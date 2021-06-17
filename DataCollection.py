import os
import urllib.parse
import re
from pathlib import Path
import itertools as it
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from tqdm.notebook import tqdm
from radiant_mlhub import client, get_session

items_pattern = re.compile(r'^/mlhub/v1/collections/(\w+)/items/(\w+)$')

def filter_item(item, classes=None, cloud_and_shadow=None, seasonal_snow=None):
    item_labels = item['properties'].get('labels',[])
    if classes is not None and not any(label in classes for label in item_labels):
        return False

    item_cloud_and_shadow = item['properties'].get('cloud_and_shadow','false') == 'true'
    if cloud_and_shadow is not None and item_cloud_and_shadow != cloud_and_shadow:
        return False

    item_seasonal_snow = item['properties'].get('seasonal_snow','false') == 'true'
    if seasonal_snow is not None and item_seasonal_snow != seasonal_snow:
        return False

    return True

def get_items(collection_id, classes=None, cloud_and_shadow=None, seasonal_snow=None, max_items=1):
    filter_fn = partial(filter_item, classes=classes, cloud_and_shadow=cloud_and_shadow, seasonal_snow=seasonal_snow)
    filtered=filter(filter_fn, client.list_collection_items(collection_id, limit=None))
    
    yield from it.islice(filtered, max_items)

def download(item, asset_key, output_dir='./data'):
    asset = item.get('assets', {}).get(asset_key)
    if asset is None:
        print(f'Asset "{asset_key}" does not exist in this item')
        return None
    
    download_url = asset.get('href')
    if download_url is None:
        print(f'Asset {asset_key} does not have an "href" property, cannot download.')
        return None
    
    session = get_session()
    r = session.get(download_url, allow_redirects=True, stream=True)
    
    filename = urllib.parse.urlsplit(r.url).path.split('/')[-1]
    output_path = Path(output_dir) / filename

    
    with output_path.open('wb') as dst:
        for chunk in r.iter_content(chunk_size=512 * 1024):
            if chunk:
                dst.write(chunk)

def download_labels_and_source(item, assets=None, output_dir='./data'):
    
    def _get_download_args(link):
        source_item_path = urllib.parse.urlsplit(link['href']).path
        source_item_collection, source_item_id = items_pattern.fullmatch(source_item_path).groups()
        source_item = client.get_collection_item(source_item_collection, source_item_id)

        source_download_dir = download_dir / 'source'
        source_download_dir.mkdir(exist_ok=True)
        
        matching_source_assets = [
            asset 
            for asset in source_item.get('assets', {}) 
            if assets is None or asset in assets
        ] 
        return [
            (source_item, asset, source_download_dir) 
            for asset in matching_source_assets
        ]

    
    download_args = []
    
    download_dir = Path(output_dir) / item['id']
    download_dir.mkdir(parents=True, exist_ok=True)
    
    labels_download_dir = download_dir / 'labels'
    labels_download_dir.mkdir(exist_ok=True)

    matching_assets = [
        asset 
        for asset in item.get('assets', {}) 
        if assets is None or asset in assets
    ]

    for asset in matching_assets:
        download_args.append((item, asset, labels_download_dir))
        
    source_links = [link for link in item['links'] if link['rel'] == 'source']
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        for argument_batch in executor.map(_get_download_args, source_links):
            download_args += argument_batch
        
    print(f'Downloading {len(download_args)} assets...')
    with ThreadPoolExecutor(max_workers=16) as executor:
        with tqdm(total=len(download_args)) as pbar:
            for _ in executor.map(lambda triplet: download(*triplet), download_args):
                pbar.update(1)