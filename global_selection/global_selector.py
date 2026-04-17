import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
load_dotenv()
import asyncio
import json
from pathlib import Path
from time import perf_counter, time
from typing import Literal
from neo4j import AsyncDriver, AsyncGraphDatabase
from async_lru import alru_cache

from openai import AsyncOpenAI
from pydantic import BaseModel

try:
    from graphiti_core.llm_client import LLMClient, LLMConfig, OpenAIClient
except ImportError:
    from graphiti_core.llm_client.client import LLMClient, OpenAIClient
    from graphiti_core.llm_client.config import LLMConfig

try:
    from graphiti_core.llm_client.config import ModelSize
except ImportError:
    try:
        from graphiti_core.llm_client import ModelSize
    except ImportError:
        from graphiti_core.models import ModelSize

try:
    from graphiti_core.prompts.models import Message
except ImportError:
    try:
        from graphiti_core.prompts import Message
    except ImportError:
        Message = dict

from mnemis_build.instrumentation import InstrumentationRecorder
from .prompts import NODE_SELECTION_PROMPT_TEMPLATE

CACHE_SIZE_PER_QUERY = 500

class NodeSelection(BaseModel):
    name: str
    uuid: str
    get_all_children: bool

class NodeSelectionList(BaseModel):
    selections: list[NodeSelection]

class Query:
    GET_MAX_LAYER = """
    match (n:Category) where n.group_id = $group_id
    return max(n.layer) as max_layer
    """
    
    GET_NODES_BY_LAYER = """
    match (n:{label}) where n.group_id = $group_id
    return n.uuid as uuid, n.name as name, n.tag as tag, n.summary as summary
    """
    
    GET_CHILD_NODES = """
    match (parent:Category)-[:CATEGORIZES]->(child:Category|Entity) where parent.group_id = $group_id and parent.uuid = $parent_uuid
    return distinct child.uuid as uuid, child.name as name, child.tag as tag, child.summary as summary, child.layer as layer
    """
    
    GET_CHILD_NODES_BATCH = """
    match (parent:Category)-[:CATEGORIZES]->(child:Category|Entity) where parent.group_id = $group_id and parent.uuid in $parent_uuids
    return distinct child.uuid as uuid, child.name as name, child.tag as tag, child.summary as summary, child.layer as layer
    """
    
    GET_ALL_DESCENDANTS = """
    match (parent:Category|Entity)-[:CATEGORIZES*1..]->(child:Category|Entity) where parent.group_id = $group_id and parent.uuid = $parent_uuid
    return distinct child.uuid as uuid, child.name as name, child.tag as tag, child.summary as summary
    """
    
    GET_ALL_DESCENDANTS_BATCH = """
    match (parent:Category|Entity)-[:CATEGORIZES*1..]->(child:Category|Entity) where parent.group_id = $group_id and parent.uuid in $parent_uuids
    return distinct child.uuid as uuid, child.name as name, child.tag as tag, child.summary as summary
    """
    
    GET_ONE_HOP_EPISODES = """
    match (n)-[:MENTIONS]-(m:Episodic)
    where n.group_id = $group_id and n.uuid = $node_uuid
    return distinct m.uuid as uuid, m.content as content, m.valid_at as valid_at
    """
    
    GET_ONE_HOP_EPISODES_BATCH = """
    match (n)-[:MENTIONS]-(m:Episodic)
    where n.group_id = $group_id and n.uuid in $node_uuids
    return distinct m.uuid as uuid, m.content as content, m.valid_at as valid_at
    """

    GET_ONE_HOP_NODES_AND_EDGES = """
    match (n)-[r:RELATES_TO]-(m:Entity)
    where n.group_id = $group_id and n.uuid = $node_uuid
    with r, m, properties(r) as rel_props
    return r.uuid as fact_uuid, r.fact as fact, r.valid_at as valid_at, rel_props['invalid_at'] as invalid_at, m.uuid as entity_uuid, m.name as name, m.tag as tag, m.summary as summary
    """
    
    GET_ONE_HOP_NODES_AND_EDGES_BATCH = """
    match (n)-[r:RELATES_TO]-(m:Entity)
    where n.group_id = $group_id and n.uuid in $node_uuids
    with r, m, properties(r) as rel_props
    return r.uuid as fact_uuid, r.fact as fact, r.valid_at as valid_at, rel_props['invalid_at'] as invalid_at, m.uuid as entity_uuid, m.name as name, m.tag as tag, m.summary as summary
    """

class GlobalSelectorConfig(BaseModel):
    use_summary: bool = False
    use_tag: bool = True
    selection_model_size: str = "large"


def _resolve_model_size(value: str | None):
    normalized = (value or "large").strip().lower()
    aliases = {
        "default": "large",
        "lg": "large",
        "md": "medium",
        "sm": "small",
    }
    normalized = aliases.get(normalized, normalized)
    members = getattr(ModelSize, "__members__", None)
    if isinstance(members, dict) and normalized in members:
        return members[normalized]
    for member_name in dir(ModelSize):
        candidate = getattr(ModelSize, member_name)
        if member_name.lower() == normalized:
            return candidate
    fallback = getattr(ModelSize, "large", None)
    if fallback is not None:
        return fallback
    if isinstance(members, dict) and members:
        return next(iter(members.values()))
    raise ValueError(f"Unsupported ModelSize value: {value!r}")


class InstrumentedGraphitiLLMClient:
    def __init__(
        self,
        inner: LLMClient,
        recorder: InstrumentationRecorder,
        *,
        default_model: str | None = None,
        small_model: str | None = None,
    ):
        self.inner = inner
        self.recorder = recorder
        self.default_model = default_model
        self.small_model = small_model

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def _snapshot_token_stats(self) -> dict[str, int]:
        getter = getattr(self.inner, "get_token_stats", None)
        if not callable(getter):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        stats = getter() or {}
        prompt_tokens = int(stats.get("prompt_tokens", stats.get("input_tokens", 0)) or 0)
        completion_tokens = int(stats.get("completion_tokens", stats.get("output_tokens", 0)) or 0)
        total_tokens = int(stats.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _extract_usage_from_response(self, response: object) -> dict[str, int]:
        if isinstance(response, dict):
            usage = response.get("usage") or response.get("_usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _diff_token_stats(self, before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        delta = {
            key: max(0, int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0))
            for key in {"prompt_tokens", "completion_tokens", "total_tokens"}
        }
        if any(delta.values()):
            return delta
        return {}

    def _resolve_model_name(self, model_size) -> str | None:
        size_name = getattr(model_size, "name", str(model_size)).lower()
        if size_name == "small":
            return self.small_model or self.default_model
        return self.default_model

    async def generate_response(
        self,
        *,
        stage: str,
        operation: str,
        messages,
        response_model,
        model_size=None,
        **kwargs,
    ):
        before = self._snapshot_token_stats()
        start = perf_counter()
        response = await self.inner.generate_response(
            messages=messages,
            response_model=response_model,
            model_size=model_size,
            **kwargs,
        )
        runtime_seconds = perf_counter() - start
        usage = self._diff_token_stats(before, self._snapshot_token_stats()) or self._extract_usage_from_response(response)
        self.recorder.record_llm_call(
            stage=stage,
            operation=operation,
            runtime_seconds=runtime_seconds,
            model=self._resolve_model_name(model_size),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens"),
            metadata={"call_type": "graphiti.generate_response", "response_model": response_model.__name__},
        )
        return response


class GlobalSelector:
    def __init__(self, driver: AsyncDriver, llm_client: LLMClient, selection_config: GlobalSelectorConfig = GlobalSelectorConfig()):
        self.driver = driver
        self.llm_client = llm_client
        self.selection_config = selection_config
        self.selection_model_size = _resolve_model_size(selection_config.selection_model_size)

    def clear_cache(self):
        self.get_max_layer.cache_clear()
        self.get_nodes_by_layer.cache_clear()
        self.get_child_nodes.cache_clear()
        self.get_all_descendants.cache_clear()
        self.get_one_hop_neighbors.cache_clear()
    
    @alru_cache(maxsize=CACHE_SIZE_PER_QUERY)
    async def get_max_layer(self, group_id: str) -> int:
        result = await self.driver.execute_query(
            Query.GET_MAX_LAYER,
            group_id=group_id
        )
        record = result.records[0]
        return record['max_layer'] if record['max_layer'] is not None else 0
    
    @alru_cache(maxsize=CACHE_SIZE_PER_QUERY)
    async def get_nodes_by_layer(self, layer: int, group_id: str) -> list[dict]:
        result = await self.driver.execute_query(
            Query.GET_NODES_BY_LAYER.format(label=f'Category_{layer}' if layer > 0 else 'Entity'),
            group_id=group_id
        )
        return [dict(record) for record in result.records]

    @alru_cache(maxsize=CACHE_SIZE_PER_QUERY)
    async def get_child_nodes(self, parent_uuid: str, group_id: str) -> list[dict]:
        result = await self.driver.execute_query(
            Query.GET_CHILD_NODES,
            group_id=group_id,
            parent_uuid=parent_uuid
        )
        return [dict(record) for record in result.records]

    async def get_child_nodes_batch(self, parent_uuids: list[str], group_id: str, mode: Literal['mp', 'batch'] = 'mp') -> list[dict]:
        if mode == 'mp':
            tasks = [self.get_child_nodes(uuid, group_id=group_id) for uuid in parent_uuids]
            results = await asyncio.gather(*tasks)
            return list({item['uuid']: item for sublist in results for item in sublist}.values())
        elif mode == 'batch':
            result = await self.driver.execute_query(
                Query.GET_CHILD_NODES_BATCH,
                group_id=group_id,
                parent_uuids=parent_uuids
            )
            return [dict(record) for record in result.records]
        else:
            raise ValueError("mode must be 'mp' or 'batch'")
    
    @alru_cache(maxsize=CACHE_SIZE_PER_QUERY)
    async def get_all_descendants(self, parent_uuid: str, group_id: str) -> list[dict]:
        result = await self.driver.execute_query(
            Query.GET_ALL_DESCENDANTS,
            group_id=group_id,
            parent_uuid=parent_uuid
        )
        return [dict(record) for record in result.records]

    async def get_all_descendants_batch(self, parent_uuids: list[str], group_id: str, mode: Literal['mp', 'batch'] = 'mp') -> list[dict]:
        if mode == 'mp':
            tasks = [self.get_all_descendants(parent_uuid, group_id=group_id) for parent_uuid in parent_uuids]
            results = await asyncio.gather(*tasks)
            return list({item['uuid']: item for sublist in results for item in sublist}.values())
        elif mode == 'batch':
            result = await self.driver.execute_query(
                Query.GET_ALL_DESCENDANTS_BATCH,
                group_id=group_id,
                parent_uuids=parent_uuids
            )
            return [dict(record) for record in result.records]
        else:
            raise ValueError("mode must be 'mp' or 'batch'")
    
    def _gather_neighbors(self, results: list) -> dict:
        assert len(results) == 2
        episodes = [dict(record) for record in results[0].records]
        edges = {}
        nodes = {}
        for record in results[1].records:
            neighbor = dict(record)
            
            edges[neighbor['fact_uuid']] = {
                'fact': neighbor['fact'],
                'valid_at': neighbor['valid_at'],
                'invalid_at': neighbor['invalid_at'],
                'uuid': neighbor['fact_uuid']
            }
            
            nodes[neighbor['entity_uuid']] = {
                'uuid': neighbor['entity_uuid'],
                'name': neighbor['name'],
                'tag': neighbor['tag'],
                'summary': neighbor['summary']
            }

        return {
            'episodes': episodes,
            'edges': list(edges.values()),
            'nodes': list(nodes.values())
        }
    
    @alru_cache(maxsize=CACHE_SIZE_PER_QUERY)
    async def get_one_hop_neighbors(self, node_uuid: str, group_id: str) -> dict:
        tasks = [self.driver.execute_query(query, group_id=group_id, node_uuid=node_uuid) for query in [Query.GET_ONE_HOP_EPISODES, Query.GET_ONE_HOP_NODES_AND_EDGES]]

        ep_result, result = await asyncio.gather(*tasks)
        return self._gather_neighbors([ep_result, result])

    async def get_one_hop_neighbors_batch(self, node_uuids: list[str], group_id: str, mode: Literal['mp', 'batch'] = 'mp') -> dict:
        if mode == 'mp':
            tasks = [self.get_one_hop_neighbors(uuid, group_id=group_id) for uuid in node_uuids]
            results = await asyncio.gather(*tasks)

            episodes = list({item['uuid']: item for res in results for item in res['episodes']}.values())
            edges = list({item['uuid']: item for res in results for item in res['edges']}.values())
            nodes = list({item['uuid']: item for res in results for item in res['nodes']}.values())

            return {
                'episodes': episodes,
                'edges': edges,
                'nodes': nodes
            }
        elif mode == 'batch':
            tasks = [self.driver.execute_query(query, group_id=group_id, node_uuids=node_uuids) for query in [Query.GET_ONE_HOP_EPISODES_BATCH, Query.GET_ONE_HOP_NODES_AND_EDGES_BATCH]]
            results = await asyncio.gather(*tasks)
            return self._gather_neighbors(results)
        else:
            raise ValueError("mode must be 'mp' or 'batch'")

    async def layer_selection(self, query: str, current_layer_categories: list[dict]) -> tuple[list[dict], list[dict]]:
        allowed_fields = [
            "uuid",
            "name",
            *(['tag'] if self.selection_config.use_tag else []),
            *(['summary'] if self.selection_config.use_summary else []),
        ]
        category_dict = {cat['uuid']: cat for cat in current_layer_categories}
        category_name_dict = {cat['name']: cat for cat in current_layer_categories}
        category_context = [{
            field: cat[field] for field in allowed_fields
        } for cat in current_layer_categories]
        
        prompt = NODE_SELECTION_PROMPT_TEMPLATE.format(
            query=query,
            nodes_info='\n'.join([json.dumps(cat, ensure_ascii=False) for cat in category_context])
        )
        response = await self.llm_client.generate_response(
            stage="global_selection",
            operation="layer_selection",
            messages=[Message(role='user', content=prompt)],
            response_model=NodeSelectionList,
            model_size=self.selection_model_size,
        )

        selected_categories = []
        shortcut_categories = []
        for selection in response.get('selections', []):
            name = selection['name']
            uuid = selection['uuid']
            get_all_children = selection['get_all_children']
            
            if uuid in category_dict:
                selected_nodes = category_dict[uuid]
            elif name in category_name_dict:
                selected_nodes = category_name_dict[name]
            else:
                print(f"Warning: LLM returned name '{name}' and uuid '{uuid}' that do not match any input node. Skipping.")
                continue
            if get_all_children:
                shortcut_categories.append(selected_nodes)
            else:
                selected_categories.append(selected_nodes)
        return selected_categories, shortcut_categories

    async def global_selection(self, query: str, group_id: str) -> tuple[dict, dict]:
        recorder = getattr(self.llm_client, "recorder", None)
        overall_start = time()
        start = time()
        mode = 'batch'
        time_stats = {}
        max_layer = await self.get_max_layer(group_id=group_id)
        time_stats['init'] = time() - start
        if recorder:
            recorder.record_stage_runtime(
                stage="global_selection",
                operation="init",
                runtime_seconds=time_stats['init'],
                metadata={"group_id": group_id},
            )

        start = time()
        previous_layer_categories = []
        selected_categories = {}
        for layer in range(max_layer, 0, -1):
            if layer == max_layer:
                current_layer_categories = await self.get_nodes_by_layer(layer, group_id=group_id)
            elif len(previous_layer_categories) > 0:
                current_layer_categories = await self.get_child_nodes_batch([cat['uuid'] for cat in previous_layer_categories], group_id=group_id, mode=mode)
            else:
                break

            selected, shortcuts = await self.layer_selection(query, current_layer_categories)
            
            for cat in selected:
                selected_categories[cat['uuid']] = cat
            all_descendants = await self.get_all_descendants_batch([cat['uuid'] for cat in shortcuts], group_id=group_id, mode=mode)
            for cat in all_descendants:
                selected_categories[cat['uuid']] = cat
            
            # print('Current Layer:', layer, 'Current Nodes:', len(current_layer_categories), 'Selected:', len(selected), 'Shortcuts:', len(shortcuts), 'All Selected:', len(selected_categories))
            previous_layer_categories = selected

        time_stats['layer_selection'] = time() - start
        if recorder:
            recorder.record_stage_runtime(
                stage="global_selection",
                operation="layer_selection_total",
                runtime_seconds=time_stats['layer_selection'],
                metadata={"group_id": group_id, "max_layer": max_layer},
            )
        
        start = time()
        neighbors = await self.get_one_hop_neighbors_batch([cat['uuid'] for cat in selected_categories.values()], group_id=group_id, mode=mode)
        time_stats['one_hop_neighbors'] = time() - start
        if recorder:
            recorder.record_stage_runtime(
                stage="global_selection",
                operation="one_hop_neighbors",
                runtime_seconds=time_stats['one_hop_neighbors'],
                metadata={"group_id": group_id, "selected_node_count": len(selected_categories)},
            )

        def format(item: dict):
            if item.get('valid_at') and type(item['valid_at']) != str:
                    item['valid_at'] = item['valid_at'].strftime("%Y/%m/%d (%a) %H:%M")
            if item.get('invalid_at') and type(item['invalid_at']) != str:
                item['invalid_at'] = item['invalid_at'].strftime("%Y/%m/%d (%a) %H:%M")
            return item
        
        episodes = [format(ep) for ep in neighbors['episodes']]
        edges = [format(edge) for edge in neighbors['edges']]
        selected_categories.update({ent['uuid']: ent for ent in neighbors['nodes']})
        nodes = [format(node) for node in selected_categories.values()]
        
        results = {
            'episodes': episodes,
            'edges': edges,
            'nodes': nodes,
        }
        if recorder:
            recorder.record_stage_runtime(
                stage="global_selection",
                operation="global_selection_total",
                runtime_seconds=time() - overall_start,
                metadata={"group_id": group_id, "query": query},
            )
        return results, time_stats

def load_locomo_data_query_group_id(file_path, group_id_prefix='locomo_ziyu', excluded_categories=None):
    """
    Load the locomo data from a JSON file.
    
    Args:
        file_path (str): The path to the JSON file containing locomo data.
        excluded_categories (set[int] | None): QA categories to skip.
        
    Returns:
        dict: The locomo data as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    query_groupid_list = []
    excluded_categories = set(excluded_categories or [])
    
    for user_id, user_data in enumerate(data):
        for query_data in user_data['qa']:
            if query_data.get('category') in excluded_categories:
                continue
            query_groupid_list.append({
                'group_id': f"{group_id_prefix}_{user_id}",
                'query': query_data['question'],
                'category': query_data.get('category')
            })
    if excluded_categories:
        print(
            f"Loaded {len(query_groupid_list)} queries from locomo data "
            f"(excluded categories: {sorted(excluded_categories)})."
        )
    else:
        print(f"Loaded {len(query_groupid_list)} queries from locomo data.")
    return query_groupid_list

def load_lme_data_query_group_id(file_path, group_id_prefix='lme_s_ziyu'):
    """
    Load the lme data from a JSON file.
    
    Args:
        file_path (str): The path to the JSON file containing lme data.
        group_id_prefix (str): The prefix to use for the group IDs.
    Returns:
        dict: The lme data as a dictionary.
    """
    data = pd.read_json(file_path).to_dict(orient='records')
    
    query_groupid_list = []
    for user_id, user_data in enumerate(data):
        query_groupid_list.append({
            'group_id': f"{group_id_prefix}_{user_id}",
            'query': user_data['question'],
            'question_id': user_data['question_id']
        })     
    print(f"Loaded {len(query_groupid_list)} queries from lme data.")
    return query_groupid_list

async def get_global_search_context(query_groupid_list, global_searcher: GlobalSelector, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def search_with_semaphore(query_data):
        async with semaphore:
            group_id = query_data['group_id']
            query = query_data['query']
            context, time_stats = await global_searcher.global_selection(query, group_id=group_id)
            return {
                "query": query,
                "group_id": group_id,
                "context": context,
                "time_stats": time_stats
            }
    # Create tasks for all queries
    tasks = [search_with_semaphore(query_data) for query_data in query_groupid_list]

    # Execute all tasks concurrently with semaphore control
    search_results = await asyncio.gather(*tasks)
    return search_results

async def parse_locomo(selector: GlobalSelector):
    group_id_prefix = os.getenv('MNEMIS_LOCOMO_GROUP_PREFIX', 'locomo_mnemis_coreAI_tel_b20_nec_full')
    max_concurrent = int(os.getenv('MNEMIS_LOCOMO_MAX_CONCURRENT', '10'))
    batch_size = int(os.getenv('MNEMIS_LOCOMO_BATCH_SIZE', '50'))
    data_path = os.getenv('MNEMIS_LOCOMO_DATA', 'data/locomo.json')
    output_path = os.getenv('MNEMIS_LOCOMO_OUTPUT', 'results/v2_locomo_mnemis_coreAI_tel_b20_nec_full.json')
    excluded_categories_env = os.getenv('MNEMIS_LOCOMO_EXCLUDE_CATEGORIES', '5')
    excluded_categories = {
        int(part.strip())
        for part in excluded_categories_env.split(',')
        if part.strip()
    }

    query_groupid_list = load_locomo_data_query_group_id(
        data_path,
        group_id_prefix=group_id_prefix,
        excluded_categories=excluded_categories,
    )
    all_data_count = len(query_groupid_list)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for i in tqdm(range(0, all_data_count, batch_size), desc=f"Processing Batches (locomo, b={batch_size})"):
            batch = query_groupid_list[i:i + batch_size]
            
            # Run the global search context retrieval
            search_results = await get_global_search_context(batch, selector, max_concurrent=max_concurrent)
            
            # Save the results to the output file
            for result in search_results:
                print(json.dumps(result, ensure_ascii=False), file=output_file)
            print(f"Processed {len(batch)} queries, saving results...")
            print(f"Batch {i // batch_size + 1} results saved to {output_path}")

async def parse_lme(selector: GlobalSelector):
    group_id_prefix = os.getenv('MNEMIS_LME_GROUP_PREFIX', 'lme_s_mnemis_coreAI_tel_b20_nec_full')
    max_concurrent = int(os.getenv('MNEMIS_LME_MAX_CONCURRENT', '10'))
    batch_size = int(os.getenv('MNEMIS_LME_BATCH_SIZE', '30'))
    data_path = os.getenv('MNEMIS_LME_DATA', 'data/longmemeval_s.json')
    output_path = os.getenv('MNEMIS_LME_OUTPUT', 'results/v2_lme_s_mnemis_coreAI_tel_b20_nec_full.json')

    query_groupid_list = load_lme_data_query_group_id(data_path, group_id_prefix=group_id_prefix)
    all_data_count = len(query_groupid_list)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for i in tqdm(range(0, all_data_count, batch_size), desc=f"Processing Batches (lme_s, b={batch_size})"):
            batch = query_groupid_list[i:i + batch_size]
            
            # Run the global search context retrieval
            search_results = await get_global_search_context(batch, selector, max_concurrent=max_concurrent)
            
            # Save the results to the output file
            for result in search_results:
                print(json.dumps(result, ensure_ascii=False), file=output_file)
            print(f"Processed {len(batch)} queries, saving results...")
            print(f"Batch {i // batch_size + 1} results saved to {output_path}")

async def main():
    url = os.getenv('MNEMIS_NEO4J_URL', 'bolt://localhost:7687')
    user = os.getenv('MNEMIS_NEO4J_USER')
    password = os.getenv('MNEMIS_NEO4J_PASSWORD')
    if not user or not password:
        raise RuntimeError("Missing Neo4j credentials. Set MNEMIS_NEO4J_USER and MNEMIS_NEO4J_PASSWORD.")
    driver = AsyncGraphDatabase.driver(url, auth=(user, password), max_connection_pool_size=1000)

    base_url = os.getenv('MNEMIS_OPENAI_BASE_URL')
    api_key = os.getenv('MNEMIS_OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("Missing OpenAI API key. Set MNEMIS_OPENAI_API_KEY (and optionally MNEMIS_OPENAI_BASE_URL).")
    if base_url:
        raw_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    else:
        raw_client = AsyncOpenAI(api_key=api_key)
    llm_config = LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=os.getenv('MNEMIS_OPENAI_MODEL'),
        small_model=os.getenv('MNEMIS_OPENAI_SMALL_MODEL'),
    )
    recorder = InstrumentationRecorder(run_name=os.getenv('MNEMIS_GLOBAL_SELECTION_RUN_NAME', 'global_selection'))
    llm_client = InstrumentedGraphitiLLMClient(
        OpenAIClient(client=raw_client, config=llm_config),
        recorder,
        default_model=os.getenv('MNEMIS_OPENAI_MODEL'),
        small_model=os.getenv('MNEMIS_OPENAI_SMALL_MODEL'),
    )
    selector = GlobalSelector(
        driver,
        llm_client,
        selection_config=GlobalSelectorConfig(
            selection_model_size=os.getenv('MNEMIS_GLOBAL_SELECTION_MODEL_SIZE', 'large')
        ),
    )

    start = time()
    with recorder.stage_timer("global_selection", "parse_locomo"):
        await parse_locomo(selector)
    # await parse_lme(selector)
    end = time()
    print(f"Total time taken: {end - start} s")
    token_stats = selector.llm_client.get_token_stats() if hasattr(selector.llm_client, 'get_token_stats') else {}
    print(token_stats)
    report_dir = Path(os.getenv('MNEMIS_INSTRUMENTATION_DIR', 'results/instrumentation'))
    report_paths = recorder.write_reports(report_dir, stem=recorder.run_name)
    print(json.dumps({"instrumentation_reports": report_paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
