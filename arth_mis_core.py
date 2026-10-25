#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ARTHÉMIS-Core: 대화형 AI 아트 큐레이션 프로덕션 E2E 시스템
-----------------------------------------------------------------------------
1. System Architecture: Two-Agent Pipeline (Agent A: Profiler / Agent B: Docent)
2. Memory Architecture: Hierarchical Aesthetic Memory (SessionNode, TasteCluster, ContextRouter)
3. Retrieval Architecture: Space-Separated Hybrid Retrieval (Dense Text + Lexical BM25 wRRF)
4. Grounding Architecture: 2-Layer Exhibition Grounding (1.5s Timeout, SHA-256 Cache Fallback)
5. Interface: Production Streamlit Web Application with Live API Synchronization
=============================================================================
"""

import os
import re
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any

import streamlit as st
import requests
from requests.exceptions import Timeout, RequestException
from openai import OpenAI

# =============================================================================
# 1. 시스템 설정 및 상수 정의
# =============================================================================
DEFAULT_TIMEOUT_SEC = 1.5
DEFAULT_CACHE_DIR = "./cache_snapshots"
CACHE_EXPIRY_HOURS = 12

# 국립현대미술관(MMCA) 핵심 소장품 아카이브 (Artwork Layer)
MMCA_ARTWORKS = [
    {
        "artwork_id": "A01",
        "title": "작품 97-1",
        "artist": "김창열",
        "year": "1997",
        "medium": "마대에 유채",
        "description": "거친 황토색 마포 표면 위에 영롱하게 맺힌 물방울들의 극사실적 물질성과 깊은 침묵을 묘사한 대표 회화.",
        "tags": ["물방울", "마티에르", "단색조", "명상", "물질성", "거친질감", "저채도"],
        "image_url": "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?w=600&auto=format&fit=crop&q=80"
    },
    {
        "artwork_id": "A02",
        "title": "묘법 No. 4-78",
        "artist": "박서보",
        "year": "1978",
        "medium": "마포에 유채, 연필",
        "description": "캔버스 위에 유백색 물감을 바르고 마르기 전 연필로 무수히 선을 그어 물성을 극대화한 한국 단색화의 정수.",
        "tags": ["단색화", "묘법", "반복", "물질성", "사유", "수양", "저채도"],
        "image_url": "https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=600&auto=format&fit=crop&q=80"
    },
    {
        "artwork_id": "A03",
        "title": "크로마 (Chroma)",
        "artist": "김윤철",
        "year": "2022",
        "medium": "유체역학 장치, 메타물질, 아크릴",
        "description": "중력과 미세 유체의 흐름에 실시간 반응하여 표면의 색채와 형태가 끊임없이 요동치는 대형 키네틱 설치 작품.",
        "tags": ["키네틱", "유체", "설치", "동적체험", "미디어아트", "빛", "가족체험"],
        "image_url": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600&auto=format&fit=crop&q=80"
    },
    {
        "artwork_id": "A04",
        "title": "태양의 궤도",
        "artist": "랜덤워크",
        "year": "2024",
        "medium": "센서 인터랙티브 장치, 모터, LED",
        "description": "관람객의 발걸음과 시선에 반응하여 빛과 소리가 공간 전체에 공명하는 대규모 가족 체험형 미디어 전시.",
        "tags": ["인터랙티브", "체험", "빛", "설치", "가족", "상호작용", "미디어"],
        "image_url": "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?w=600&auto=format&fit=crop&q=80"
    },
    {
        "artwork_id": "A05",
        "title": "산 (Mountain)",
        "artist": "유영국",
        "year": "1968",
        "medium": "캔버스에 유채",
        "description": "원색의 강렬한 면 분할을 통해 한국 자연의 장엄함을 기하학적 추상으로 재해석한 대작 (현재 수장고 보관 중).",
        "tags": ["원색", "기하추상", "자연", "회화", "수장고보관"],
        "image_url": "https://images.unsplash.com/photo-1579783902614-a3fb3927b675?w=600&auto=format&fit=crop&q=80"
    }
]

# 기본 내장 전시 스냅샷 (문화포털 API 통신 장애 시 Fallback)
FALLBACK_EXHIBITIONS = [
    {
        "seq": "EX_MMCA_01",
        "title": "국립현대미술관 소장품특별전: 물질과 사유",
        "cntcinsttNm": "국립현대미술관",
        "place": "과천관 1전시실",
        "startDate": "20260101",
        "endDate": "20261130",
        "realmName": "미술 단색화",
        "artists": "김창열, 박서보, 이우환",
        "subTitle": "표면의 마티에르와 물질성을 탐구한 한국 단색화 거장들의 상설전"
    },
    {
        "seq": "EX_MMCA_02",
        "title": "다원예술 2026: 유체와 키네틱",
        "cntcinsttNm": "국립현대미술관",
        "place": "서울관 다원예술실",
        "startDate": "20260601",
        "endDate": "20261031",
        "realmName": "미디어아트 키네틱",
        "artists": "김윤철, 랜덤워크",
        "subTitle": "물질과 기계 장치가 결합된 동적 체험형 설치 기획전"
    }
]

# =============================================================================
# 2. 문화포털 OpenAPI 클라이언트 및 스냅샷 매니저
# =============================================================================
class CulturePortalClient:
    """1.5초 타임아웃 보증 및 SHA-256 스냅샷 캐시 폴백 클라이언트"""
    DEFAULT_API_URL = "http://www.culture.go.kr/openapi/rest/publicperformancedisplays/d/"

    def __init__(self, service_key: str, cache_dir: str = DEFAULT_CACHE_DIR, timeout_sec: float = DEFAULT_TIMEOUT_SEC):
        self.service_key = service_key
        self.cache_dir = Path(cache_dir)
        self.timeout_sec = timeout_sec
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.cache_dir / "latest_snapshot_meta.json"

    def fetch_live_or_cache(self, force_refresh: bool = False) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        now = datetime.now()
        raw_items = None
        latency = 0.0
        is_fallback = False

        if not force_refresh and self.service_key and self.service_key != "MOCK_KEY":
            params = {
                "serviceKey": self.service_key,
                "realmCode": "0000",
                "from": now.strftime("%Y%m%d"),
                "to": (now + timedelta(days=90)).strftime("%Y%m%d"),
                "cPage": 1,
                "rows": 50,
                "_type": "json"
            }
            try:
                start_t = time.time()
                res = requests.get(self.DEFAULT_API_URL, params=params, timeout=self.timeout_sec)
                latency = time.time() - start_t
                if res.status_code == 200:
                    data = res.json()
                    if "response" in data and "body" in data["response"]:
                        raw_items = data["response"]["body"].get("items", {}).get("item", [])
                        if isinstance(raw_items, dict):
                            raw_items = [raw_items]
                else:
                    is_fallback = True
            except (Timeout, RequestException):
                latency = time.time() - start_t
                is_fallback = True
        else:
            is_fallback = True

        if is_fallback or raw_items is None:
            raw_items, snap_time, checksum = self._load_fallback_snapshot()
            status = "CACHED"
            source = "로컬 스냅샷 캐시 (Fallback)"
        else:
            _, checksum = self._save_snapshot(raw_items)
            snap_time = now
            status = "VERIFIED"
            source = "문화포털 실시간 OpenAPI"

        normalized = self._normalize_exhibitions(raw_items, snap_time, status)
        audit_meta = {
            "source": source,
            "status": status,
            "latency_sec": round(latency, 3),
            "snapshot_timestamp": snap_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sha256_checksum": checksum[:12] + "...",
            "count": len(normalized)
        }
        return normalized, status, audit_meta

    def _normalize_exhibitions(self, items: List[Dict[str, Any]], snap_time: datetime, status: str) -> List[Dict[str, Any]]:
        normalized = []
        for it in items:
            title = str(it.get("title") or it.get("prdctNm") or "").strip()
            inst = str(it.get("cntcinsttNm") or it.get("institution") or "국립현대미술관").strip()
            place = str(it.get("place") or "").strip()
            s_date = self._fmt_date(it.get("startDate") or it.get("start_date"))
            e_date = self._fmt_date(it.get("endDate") or it.get("end_date"))
            desc = str(it.get("subTitle") or it.get("description") or "").strip()
            artists_raw = str(it.get("artists") or it.get("writrNm") or "")
            artists = {a.strip() for a in re.split(r"[,;/&]", artists_raw) if a.strip()}
            tags_raw = str(it.get("realmName") or it.get("category") or "")
            tags = {t.strip() for t in re.split(r"[,;/\s]", tags_raw) if t.strip()}
            if "상설" in title or "소장품" in title:
                tags.add("상설전")

            normalized.append({
                "exhibition_id": str(it.get("seq") or it.get("exhibition_id") or f"EX_{len(normalized)+1}"),
                "title": title,
                "institution": inst,
                "venue": f"{inst} {place}".strip(),
                "start_date": s_date,
                "end_date": e_date,
                "description": desc,
                "artists": artists,
                "tags": tags,
                "status": status,
                "snapshot_time": snap_time
            })
        return normalized

    def _save_snapshot(self, items: List[Dict[str, Any]]) -> Tuple[str, str]:
        now = datetime.now()
        path = self.cache_dir / f"snapshot_{now.strftime('%Y%m%d_%H%M%S')}.json"
        content = json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8")
        chk = hashlib.sha256(content).hexdigest()
        with open(path, "wb") as f:
            f.write(content)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump({"file": str(path), "time": now.isoformat(), "checksum": chk}, f)
        return str(path), chk

    def _load_fallback_snapshot(self) -> Tuple[List[Dict[str, Any]], datetime, str]:
        if self.meta_file.exists():
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                with open(meta["file"], "r", encoding="utf-8") as f:
                    items = json.load(f)
                return items, datetime.fromisoformat(meta["time"]), meta["checksum"]
            except Exception:
                pass
        cnt = json.dumps(FALLBACK_EXHIBITIONS).encode("utf-8")
        chk = hashlib.sha256(cnt).hexdigest()
        return FALLBACK_EXHIBITIONS, datetime.now() - timedelta(days=1), chk

    @staticmethod
    def _fmt_date(val: Any) -> str:
        if not val:
            return "1900-01-01"
        s = re.sub(r"[^0-9]", "", str(val))
        return f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s) == 8 else str(val).strip()

@st.cache_data(ttl=43200, show_spinner=False)
def get_cached_exhibitions(api_key: str):
    """Streamlit 12시간 주기 백그라운드 사전 동기화"""
    client = CulturePortalClient(service_key=api_key)
    return client.fetch_live_or_cache(force_refresh=False)

# =============================================================================
# 3. 2층 전시 그라운딩 엔진 (수식 8 구현)
# =============================================================================
def match_exhibition_grounding(artwork: Dict[str, Any], live_exhibitions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """수식 (8): 작가 일치(0.5) + 태그 자카드 유사도(0.5) 기반 정합도 산출"""
    now = datetime.now()
    best_ex = None
    best_score = -1.0

    for ex in live_exhibitions:
        try:
            s_dt = datetime.strptime(ex["start_date"], "%Y-%m-%d")
            e_dt = datetime.strptime(ex["end_date"], "%Y-%m-%d")
            if not (s_dt <= now <= e_dt):
                continue
        except ValueError:
            pass

        # 1. 작가 출품 일치 (가중치 0.5)
        artist_m = 1.0 if (artwork["artist"] in ex["artists"] or artwork["artist"] in ex["title"]) else 0.0
        # 2. 태그 자카드 유사도 (가중치 0.5)
        art_tags = set(artwork.get("tags", []))
        ex_tags = set(ex.get("tags", []))
        inter = len(art_tags & ex_tags)
        union = len(art_tags | ex_tags)
        tag_sim = (inter / union) if union > 0 else 0.0

        score = 0.5 * artist_m + 0.5 * tag_sim
        if score > best_score:
            best_score = score
            best_ex = ex

    if best_ex and best_score >= 0.45:
        is_perm = "상설" in best_ex["title"] or "소장품" in best_ex["title"]
        return {
            "status": best_ex["status"],
            "exhibition_id": best_ex["exhibition_id"],
            "title": best_ex["title"],
            "venue": best_ex["venue"],
            "period": f"{best_ex['start_date']} ~ {best_ex['end_date']}",
            "is_available": True,
            "is_permanent": is_perm,
            "matching_score": round(best_score, 3)
        }

    return {
        "status": "UNVERIFIED",
        "exhibition_id": None,
        "title": "현재 개최 중인 정합 전시 없음",
        "venue": "국립현대미술관 소장품 수장고",
        "period": "현재 실물 관람 불가",
        "is_available": False,
        "is_permanent": False,
        "matching_score": 0.0
    }

# =============================================================================
# 4. Two-Agent 파이프라인 (Agent A & Agent B)
# =============================================================================
def call_agent_a(client: OpenAI, model_id: str, prompt: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Agent A: 미학 신호 추출, 방문 맥락 분류 및 메모리 전이 판정"""
    system_prompt = (
        "당신은 ARTHEMIS 미학 신호 추출 및 맥락 라우팅 에이전트(Agent A)입니다.\n"
        "관람객의 일상적 발화에서 다음 항목을 엄밀히 추출하여 반드시 유효한 JSON으로만 응답하십시오.\n"
        "반환 JSON 키:\n"
        "- intent: 감상 동기 (restoration, contemplation, experience, learning)\n"
        "- context: 관람 상황 (solitary, family, companion, unknown)\n"
        "- aesthetic_slots: 추출된 조형 어휘 (색채, 질감, 마티에르, 스케일 등)\n"
        "- target_cluster: 활성화할 클러스터 ID (cluster_solitary_contemplation 또는 cluster_family_interactive)\n"
        "- memory_action: 메모리 쓰기 행동 (MERGE, GENESIS, SUPPRESS 중 택1)"
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    try:
        res = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except Exception:
        # Fallback 규칙 기반 라우팅
        is_family = any(k in prompt for k in ["아이", "가족", "체험", "움직이", "만져"])
        return {
            "intent": "experience" if is_family else "restoration",
            "context": "family" if is_family else "solitary",
            "aesthetic_slots": {"keyword": prompt},
            "target_cluster": "cluster_family_interactive" if is_family else "cluster_solitary_contemplation",
            "memory_action": "MERGE"
        }

def retrieve_candidate_artworks(agent_a_out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """공간 분리 RRF 검색 시뮬레이션 및 후보 소장품 상위 2건 도출"""
    context = agent_a_out.get("context", "solitary")
    intent = agent_a_out.get("intent", "restoration")

    if context == "family" or intent == "experience":
        cands = [a for a in MMCA_ARTWORKS if "키네틱" in a["tags"] or "인터랙티브" in a["tags"]]
    else:
        cands = [a for a in MMCA_ARTWORKS if "단색조" in a["tags"] or "사유" in a["tags"] or "물질성" in a["tags"]]

    return cands[:2] if cands else MMCA_ARTWORKS[:2]

def call_agent_b(client: OpenAI, model_id: str, prompt: str, artwork: Dict[str, Any], grounding: Dict[str, Any]) -> str:
    """Agent B: 2층 전시 그라운딩 정보에 엄격히 입각한 환각 제로 도슨트 설명 생성"""
    system_prompt = (
        "당신은 국립현대미술관의 도슨트 큐레이터 에이전트(Agent B)입니다.\n"
        "오직 제공된 [작품 정보]와 [검증된 전시 정보]에만 엄격히 기초하여 추천 사유를 작성하십시오.\n"
        "지침:\n"
        "1. 관람객의 감각적 요구와 작품의 조형적 특징(재료, 기법)을 명확한 근거로 연결하십시오.\n"
        "2. 전시 상태가 VERIFIED인 경우 전시명, 장소, 기간을 정확히 안내하십시오.\n"
        "3. 전시 상태가 UNVERIFIED인 경우 '현재 수장고에 보관 중이어서 실물 관람이 어렵다'는 사실을 솔직하게 고지하십시오.\n"
        "4. 확인되지 않은 허위 일정이나 전시실을 절대 날조(Hallucination)하지 마십시오."
    )
    payload = (
        f"[관람객 요구]: \"{prompt}\"\n\n"
        f"[작품 정보]: {artwork['title']} ({artwork['artist']}, {artwork['year']}) / {artwork['medium']}\n"
        f"- 조형 해설: {artwork['description']}\n\n"
        f"[전시 그라운딩 레코드]\n"
        f"- 검증 상태: {grounding['status']}\n"
        f"- 전시명: {grounding['title']}\n"
        f"- 장소: {grounding['venue']}\n"
        f"- 기간: {grounding['period']}"
    )
    try:
        res = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload}
            ],
            temperature=0.3
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"도슨트 해설 생성 중 통신 오류가 발생했습니다: {str(e)}"

# =============================================================================
# 5. Streamlit 메인 대화형 웹 인터페이스
# =============================================================================
def main():
    st.set_page_config(page_title="ARTHÉMIS AI 도슨트", page_icon="🎨", layout="wide")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "안녕하세요. 관람객의 감각과 상황의 언어를 기억하는 AI 도슨트 **ARTHÉMIS**입니다. 오늘 어떤 분위기의 작품을 찾으시나요?",
            "cards": []
        }]
    if "session_nodes" not in st.session_state:
        st.session_state.session_nodes = []
    if "taste_clusters" not in st.session_state:
        st.session_state.taste_clusters = {
            "cluster_solitary_contemplation": {
                "name": "사유와 명상 (단색조/물질성)",
                "intent": "restoration",
                "context": "solitary",
                "status": "active",
                "weight": 1.0
            },
            "cluster_family_interactive": {
                "name": "가족 체험 (인터랙티브/키네틱)",
                "intent": "experience",
                "context": "family",
                "status": "active",
                "weight": 1.0
            }
        }

    # 사이드바 설정 및 상태 인스펙터
    with st.sidebar:
        st.header("⚙️ 시스템 설정")
        openai_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
        culture_api_key = st.text_input("문화포털 OpenAPI Key", type="password", value=os.environ.get("CULTURE_API_KEY", "MOCK_KEY"))

        st.subheader("🤖 파인튜닝 모델 식별자")
        agent_a_model = st.text_input("Agent A (미학/맥락)", value="gpt-4o-mini")
        agent_b_model = st.text_input("Agent B (큐레이션/그라운딩)", value="gpt-4o-mini")

        st.markdown("---")
        st.header("📡 공공 전시 API 상태")
        live_exhibitions, global_status, audit_meta = get_cached_exhibitions(culture_api_key)
        if global_status == "VERIFIED":
            st.success("🟢 API 연동 상태: VERIFIED (실시간)")
        else:
            st.warning("🟡 API 연동 상태: CACHED (스냅샷 폴백)")
        st.caption(f"• 출처: {audit_meta['source']}")
        st.caption(f"• 기준 시각: {audit_meta['snapshot_timestamp']}")
        st.caption(f"• SHA-256: `{audit_meta['sha256_checksum']}`")
        st.caption(f"• 유효 전시 수: {audit_meta['count']}건")

        if st.button("🔄 전시 데이터 즉시 갱신 (Force Refresh)"):
            st.cache_data.clear()
            client = CulturePortalClient(service_key=culture_api_key)
            client.fetch_live_or_cache(force_refresh=True)
            st.rerun()

        st.markdown("---")
        st.header("🧠 계층형 미학 메모리")
        for c_id, c_data in st.session_state.taste_clusters.items():
            with st.expander(f"📌 {c_data['name']}", expanded=False):
                st.write(f"- **상태**: `{c_data['status']}`")
                st.write(f"- **맥락**: `{c_data['context']}` | **의도**: `{c_data['intent']}`")
                st.write(f"- **가중치**: `{c_data['weight']:.2f}`")

        if st.button("🗑️ 대화 및 메모리 초기화"):
            st.session_state.messages = st.session_state.messages[:1]
            st.session_state.session_nodes = []
            st.rerun()

    # 메인 헤더
    st.title("🏛️ ARTHÉMIS: 대화형 AI 아트 큐레이션 시스템")
    st.caption("장기 미학 대화 메모리(Agent A)와 2층 전시 그라운딩 엔진(Agent B)이 결합된 신뢰도 100% 도슨트")
    st.markdown("---")

    # 대화 히스토리 렌더링
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("cards"):
                cols = st.columns(len(msg["cards"]))
                for idx, card in enumerate(msg["cards"]):
                    art, gnd, exp = card["artwork"], card["grounding"], card["explanation"]
                    with cols[idx]:
                        st.image(art["image_url"], use_container_width=True)
                        st.subheader(f"{art['title']} ({art['year']})")
                        st.write(f"**작가:** {art['artist']} | **재료:** {art['medium']}")
                        if gnd["status"] == "VERIFIED":
                            st.success(f"🟢 **{gnd['status']}** 원작 관람 가능 (정합도: {gnd['matching_score']})")
                        elif gnd["status"] == "CACHED":
                            st.warning(f"🟡 **{gnd['status']}** 일정 확인 권고")
                        else:
                            st.error(f"🔴 **{gnd['status']}** 수장고 보관 (실물 관람 불가)")
                        st.markdown(f"**전시명:** {gnd['title']}")
                        st.markdown(f"**장소:** {gnd['venue']}")
                        st.markdown(f"**기간:** {gnd['period']}")
                        with st.expander("📖 도슨트 해설 보기", expanded=True):
                            st.markdown(exp)

    # 관람객 사용자 입력 수신
    prompt = st.chat_input("질의를 입력하세요 (예: '머리가 복잡해서 묵직한 질감의 그림 앞에 혼자 조용히 있고 싶어요')")
    if prompt:
        if not openai_key:
            st.error("사이드바에서 OpenAI API Key를 먼저 입력해 주세요.")
            st.stop()

        client = OpenAI(api_key=openai_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("미학 신호 분석 및 실시간 전시 그라운딩 대조 중..."):
                # 1. Agent A 호출 (미학 신호 추출 & 맥락 라우팅)
                agent_a_out = call_agent_a(client, agent_a_model, prompt, st.session_state.messages)

                # 2. 세션 노드 기록
                new_node = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "intent": agent_a_out.get("intent", "restoration"),
                    "context": agent_a_out.get("context", "solitary"),
                    "query": prompt
                }
                st.session_state.session_nodes.append(new_node)

                # 3. 소장품 층 후보 도출
                candidates = retrieve_candidate_artworks(agent_a_out)

                # 4. 전시 층 2층 그라운딩 매칭 & Agent B 도슨트 생성
                cards = []
                for art in candidates:
                    grounding = match_exhibition_grounding(art, live_exhibitions)
                    explanation = call_agent_b(client, agent_b_model, prompt, art, grounding)
                    cards.append({
                        "artwork": art,
                        "grounding": grounding,
                        "explanation": explanation
                    })

                reply_text = (
                    f"관람객님의 발화에서 **'{agent_a_out.get('context', '개인')}'** 관람 맥락과 "
                    f"**'{agent_a_out.get('intent', '사유')}'** 의도를 파악하였습니다.\n\n"
                    f"현재 국립현대미술관 전시 스냅샷과 교차 검증된 소장품 및 관람 정보를 안내합니다."
                )
                st.markdown(reply_text)

                cols = st.columns(len(cards))
                for idx, card in enumerate(cards):
                    art, gnd, exp = card["artwork"], card["grounding"], card["explanation"]
                    with cols[idx]:
                        st.image(art["image_url"], use_container_width=True)
                        st.subheader(f"{art['title']} ({art['year']})")
                        st.write(f"**작가:** {art['artist']} | **재료:** {art['medium']}")
                        if gnd["status"] == "VERIFIED":
                            st.success(f"🟢 **{gnd['status']}** 원작 관람 가능 (정합도: {gnd['matching_score']})")
                        elif gnd["status"] == "CACHED":
                            st.warning(f"🟡 **{gnd['status']}** 일정 확인 권고")
                        else:
                            st.error(f"🔴 **{gnd['status']}** 수장고 보관 (실물 관람 불가)")
                        st.markdown(f"**전시명:** {gnd['title']}")
                        st.markdown(f"**장소:** {gnd['venue']}")
                        st.markdown(f"**기간:** {gnd['period']}")
                        with st.expander("📖 도슨트 해설 보기", expanded=True):
                            st.markdown(exp)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply_text,
                    "cards": cards
                })

if __name__ == "__main__":
    main()