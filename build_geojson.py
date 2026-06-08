# -*- coding: utf-8 -*-
"""maps-main.zip의 시군구 GeoJSON을 병합/단순화하여 data/sigungu_merged.geojson 생성."""
import zipfile, json, os
from shapely.geometry import shape, mapping, MultiPolygon
from shapely.ops import transform as shp_transform
from shapely import get_coordinates
from pyproj import Transformer

# 원본 좌표계: EPSG:5179 (Korea UTM-K) -> WGS84(EPSG:4326)
_TF = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)


def _reproject(geom):
    return shp_transform(lambda x, y, z=None: _TF.transform(x, y), geom)

SIDO_SHORT = {
    '서울특별시': '서울', '부산광역시': '부산', '대구광역시': '대구', '인천광역시': '인천',
    '광주광역시': '광주', '대전광역시': '대전', '울산광역시': '울산', '세종특별자치시': '세종',
    '세종시': '세종', '경기도': '경기', '강원도': '강원', '강원특별자치도': '강원',
    '충청북도': '충북', '충청남도': '충남', '전라북도': '전북', '전북특별자치도': '전북',
    '전라남도': '전남', '경상북도': '경북', '경상남도': '경남', '제주특별자치도': '제주', '제주도': '제주',
}


def short(s):
    s = str(s).strip()
    return SIDO_SHORT.get(s, s)


def rc(o, nd=3):
    if isinstance(o, list):
        if o and isinstance(o[0], (int, float)):
            return [round(float(o[0]), nd), round(float(o[1]), nd)]
        return [rc(x, nd) for x in o]
    return o


def drop_small(geom, min_area):
    """면적이 작은 폴리곤(작은 섬) 제거 -> 용량 대폭 절감."""
    if isinstance(geom, MultiPolygon):
        parts = [g for g in geom.geoms if g.area >= min_area]
        if not parts:
            parts = [max(geom.geoms, key=lambda g: g.area)]
        return MultiPolygon(parts) if len(parts) > 1 else parts[0]
    return geom


def fix(n):
    for enc in ('cp437', 'latin-1'):
        try:
            return n.encode(enc).decode('cp949')
        except Exception:
            pass
    return n


def main(zip_path='maps-main.zip', out='data/sigungu_merged.geojson', tol=0.0025, min_area=0.0003):
    z = zipfile.ZipFile(zip_path)
    merged = {"type": "FeatureCollection", "features": []}
    tot = 0
    for n in z.namelist():
        kor = fix(n)
        if n.lower().endswith('.json') and '/json/' in n and '시군구' in kor:
            ss = short(kor.split('/json/')[-1].split('_')[0])
            data = json.loads(z.read(n).decode('utf-8'))
            for f in data['features']:
                title = str(f['properties'].get('title', '')).strip()
                g = _reproject(shape(f['geometry']))          # EPSG:5179 -> WGS84(degrees)
                g = drop_small(g, min_area)                    # 작은 섬 제거(deg²)
                g = g.simplify(tol, preserve_topology=False)   # 정점 단순화(deg)
                if g.is_empty:
                    g = drop_small(_reproject(shape(f['geometry'])), min_area)
                gj = mapping(g)
                tot += len(get_coordinates(g))
                merged['features'].append({
                    'type': 'Feature',
                    'properties': {'join_key': f"{ss}_{title}", 'sido': ss, 'sigungu': title},
                    'geometry': {'type': gj['type'], 'coordinates': rc(gj['coordinates'], 4)},
                })
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fp:
        json.dump(merged, fp, ensure_ascii=False, separators=(',', ':'))
    print('features:', len(merged['features']), '| total pts:', tot,
          '| size MB:', round(os.path.getsize(out) / 1e6, 2))


if __name__ == '__main__':
    main()
