#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
잡코리아 LLM Engineer AI Challenge - 과제 1,2 추론 코드
LLM 응답 지연 및 부정확한 응답 문제 해결을 위한 통합 시스템
"""

import asyncio
import json
import time
import logging
import redis
import numpy as np
import hashlib
import re
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
# from concurrent.futures import ThreadPoolExecutor  # 현재 사용되지 않음
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 설정
class Config:
    REDIS_URL = "redis://localhost:6379"
    OPENAI_API_KEY = "your-openai-api-key"
    ANTHROPIC_API_KEY = "your-anthropic-api-key"
    CACHE_TTL = 3600  # 1시간
    RESPONSE_TIMEOUT = 30  # 30초
    MAX_TOKENS = 2000

# 데이터 클래스
@dataclass
class JobPostingRequest:
    company_name: str
    position: str
    requirements: List[str]
    benefits: List[str]
    company_info: Dict[str, str]
    user_preference: str = "balanced"  # fast, quality, balanced

@dataclass
class JobPostingResponse:
    title: str
    description: str
    requirements: List[str]
    benefits: List[str]
    company_intro: str
    response_time: float
    model_used: str
    confidence_score: float

class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class ModelType(Enum):
    FAST = "fast"      # GPT-4o-mini
    QUALITY = "quality"  # GPT-4.1
    SPECIALIZED = "specialized"  # Claude-4.1
    BALANCED = "balanced"  # 균형잡힌 선택

# 캐시 관리자
class CacheManager:
    def __init__(self):
        self.redis_client = redis.from_url(Config.REDIS_URL)
        self.local_cache = {}
    
    def get_cache_key(self, request: JobPostingRequest) -> str:
        """요청을 기반으로 캐시 키 생성"""
        key_data = {
            "company": request.company_name,
            "position": request.position,
            "requirements": sorted(request.requirements),
            "benefits": sorted(request.benefits)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"job_posting:{hashlib.sha256(key_str.encode()).hexdigest()}"
    
    def get_cached_response(self, request: JobPostingRequest) -> Optional[JobPostingResponse]:
        """캐시된 응답 조회"""
        try:
            cache_key = self.get_cache_key(request)
            
            # 로컬 캐시 확인
            if cache_key in self.local_cache:
                cached_data = self.local_cache[cache_key]
                if time.time() - cached_data["timestamp"] < Config.CACHE_TTL:
                    logger.info(f"로컬 캐시 히트: {cache_key}")
                    return cached_data["response"]
            
            # Redis 캐시 확인
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                response_data = json.loads(cached_data)
                response = JobPostingResponse(**response_data)
                logger.info(f"Redis 캐시 히트: {cache_key}")
                return response
            
            return None
        except Exception as e:
            logger.error(f"캐시 조회 오류: {e}")
            return None
    
    def cache_response(self, request: JobPostingRequest, response: JobPostingResponse):
        """응답을 캐시에 저장"""
        try:
            cache_key = self.get_cache_key(request)
            response_data = {
                "title": response.title,
                "description": response.description,
                "requirements": response.requirements,
                "benefits": response.benefits,
                "company_intro": response.company_intro,
                "response_time": response.response_time,
                "model_used": response.model_used,
                "confidence_score": response.confidence_score
            }
            
            # Redis에 저장
            self.redis_client.setex(
                cache_key, 
                Config.CACHE_TTL, 
                json.dumps(response_data)
            )
            
            # 로컬 캐시에 저장
            self.local_cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
            
            logger.info(f"응답 캐시 저장: {cache_key}")
        except Exception as e:
            logger.error(f"캐시 저장 오류: {e}")

# 요청 복잡도 분석기
class ComplexityAnalyzer:
    def __init__(self):
        self.complexity_thresholds = {
            "simple": 50,    # 간단한 요청
            "medium": 150,   # 중간 복잡도
            "complex": 300   # 복잡한 요청
        }
    
    def analyze_complexity(self, request: JobPostingRequest) -> ComplexityLevel:
        """요청의 복잡도를 분석하여 적절한 모델 선택"""
        # 요구사항과 혜택의 복잡도 계산
        requirements_complexity = sum(len(req) for req in request.requirements)
        benefits_complexity = sum(len(benefit) for benefit in request.benefits)
        company_info_complexity = sum(len(str(v)) for v in request.company_info.values())
        
        total_complexity = requirements_complexity + benefits_complexity + company_info_complexity
        
        if total_complexity <= self.complexity_thresholds["simple"]:
            return ComplexityLevel.SIMPLE
        elif total_complexity <= self.complexity_thresholds["medium"]:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.COMPLEX

# Content Safety Filter
class ContentSafetyFilter:
    def __init__(self):
        self.bias_keywords = [
            "남성", "여성", "20대", "30대", "40대", "50대",
            "외모", "키", "몸무게", "성별", "연령"
        ]
        self.sensitive_keywords = [
            "연봉", "급여", "월급", "수당", "보너스",
            "주민번호", "주민등록번호", "주소", "전화번호"
        ]
        self.inappropriate_keywords = [
            "욕설", "비속어", "차별적", "부적절"
        ]
        
        # 정규식 패턴 추가
        self.patterns = {
            'phone': re.compile(r'(\d{2,4}[-\s]?\d{2,4}[-\s]?\d{4})'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'rrn': re.compile(r'\d{6}[-\s]?\d{7}'),  # 주민등록번호
            'address': re.compile(r'[가-힣]+시\s*[가-힣]+구\s*[가-힣]+동'),
            'salary': re.compile(r'(\d{1,3}(?:,\d{3})*)\s*(?:만원|천원|원)')
        }
    
    def filter_content(self, content: str) -> Tuple[str, float]:
        """콘텐츠를 필터링하고 위험도 점수 반환"""
        risk_score = 0.0
        filtered_content = content
        
        # 편향성 검사
        bias_count = sum(1 for keyword in self.bias_keywords if keyword in content)
        if bias_count > 0:
            risk_score += bias_count * 0.3
            filtered_content = self._remove_bias_content(filtered_content)
        
        # 민감 정보 검사
        sensitive_count = sum(1 for keyword in self.sensitive_keywords if keyword in content)
        if sensitive_count > 0:
            risk_score += sensitive_count * 0.4
            filtered_content = self._remove_sensitive_content(filtered_content)
        
        # 부적절한 내용 검사
        inappropriate_count = sum(1 for keyword in self.inappropriate_keywords if keyword in content)
        if inappropriate_count > 0:
            risk_score += inappropriate_count * 0.5
            filtered_content = self._sanitize_content(filtered_content)
        
        # 정규식 기반 민감 정보 검사
        filtered_content, regex_risk = self._apply_regex_filters(filtered_content)
        risk_score += regex_risk
        
        return filtered_content, min(risk_score, 1.0)
    
    def _remove_bias_content(self, content: str) -> str:
        """편향적 내용 제거"""
        for keyword in self.bias_keywords:
            content = content.replace(keyword, "[제거됨]")
        return content
    
    def _remove_sensitive_content(self, content: str) -> str:
        """민감한 정보 제거"""
        for keyword in self.sensitive_keywords:
            content = content.replace(keyword, "[협의 가능]")
        return content
    
    def _sanitize_content(self, content: str) -> str:
        """부적절한 내용 정리"""
        for keyword in self.inappropriate_keywords:
            content = content.replace(keyword, "[수정됨]")
        return content
    
    def _apply_regex_filters(self, content: str) -> Tuple[str, float]:
        """정규식 기반 민감 정보 필터링"""
        risk_score = 0.0
        
        # 전화번호 마스킹
        content = self.patterns['phone'].sub(r'[전화번호]', content)
        
        # 이메일 마스킹
        content = self.patterns['email'].sub('[이메일]', content)
        
        # 주민등록번호 마스킹
        content = self.patterns['rrn'].sub('[주민번호]', content)
        
        # 주소 마스킹
        content = self.patterns['address'].sub('[주소]', content)
        
        # 급여 정보 마스킹
        content = self.patterns['salary'].sub('[협의 가능]', content)
        
        # 정규식 패턴이 발견되면 위험도 증가
        if any(pattern.search(content) for pattern in self.patterns.values()):
            risk_score += 0.3
        
        return content, risk_score

# RAG 엔진
class RAGEngine:
    def __init__(self):
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.company_templates = self._load_company_templates()
        self.job_templates = self._load_job_templates()
    
    def _load_company_templates(self) -> List[str]:
        """기업 정보 템플릿 로드"""
        return [
            "우리 회사는 혁신적인 기술로 고객 가치를 창출하는 기업입니다.",
            "직원의 성장과 발전을 최우선으로 생각하는 기업 문화를 가지고 있습니다.",
            "업계 최고 수준의 복리후생과 근무 환경을 제공합니다.",
            "지속적인 학습과 자기계발을 장려하는 기업입니다."
        ]
    
    def _load_job_templates(self) -> List[str]:
        """직무별 템플릿 로드"""
        return [
            "해당 직무에 필요한 핵심 역량과 경험을 보유한 인재를 모집합니다.",
            "팀워크와 소통 능력이 뛰어난 인재를 찾고 있습니다.",
            "새로운 도전을 두려워하지 않는 창의적인 인재를 환영합니다.",
            "지속적인 성장과 발전에 관심이 많은 인재를 모집합니다."
        ]
    
    def retrieve_relevant_context(self, request: JobPostingRequest) -> str:
        """요청과 관련된 컨텍스트 검색"""
        query = f"{request.position} {request.company_name}"
        query_embedding = self.embedding_model.encode(query)
        
        # 관련 템플릿 검색
        relevant_templates = []
        
        # 기업 정보 관련성 검사
        for template in self.company_templates:
            template_embedding = self.embedding_model.encode(template)
            similarity = np.dot(query_embedding, template_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(template_embedding)
            )
            if similarity > 0.3:
                relevant_templates.append(template)
        
        # 직무 관련성 검사
        for template in self.job_templates:
            template_embedding = self.embedding_model.encode(template)
            similarity = np.dot(query_embedding, template_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(template_embedding)
            )
            if similarity > 0.3:
                relevant_templates.append(template)
        
        return "\n".join(relevant_templates)

# LLM 서비스
class LLMService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.anthropic_client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    
    async def generate_job_posting(
        self, 
        request: JobPostingRequest, 
        context: str,
        model_type: ModelType
    ) -> JobPostingResponse:
        """채용공고 생성"""
        start_time = time.time()
        
        try:
            if model_type == ModelType.FAST:
                response = await self._generate_with_fast_model(request, context)
            elif model_type == ModelType.QUALITY:
                response = await self._generate_with_quality_model(request, context)
            else:  # SPECIALIZED
                response = await self._generate_with_specialized_model(request, context)
            
            response.response_time = time.time() - start_time
            return response
            
        except Exception as e:
            logger.error(f"LLM 생성 오류: {e}")
            # 폴백 응답 생성
            return self._generate_fallback_response(request, start_time)
    
    async def _generate_with_fast_model(
        self, 
        request: JobPostingRequest, 
        context: str
    ) -> JobPostingResponse:
        """빠른 모델로 채용공고 생성"""
        prompt = self._build_prompt(request, context, "fast")
        
        response = await asyncio.to_thread(
            self.openai_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=Config.MAX_TOKENS,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        parsed_response = self._parse_llm_response(content)
        
        return JobPostingResponse(
            title=parsed_response.get("title", f"{request.position} 모집"),
            description=parsed_response.get("description", ""),
            requirements=parsed_response.get("requirements", request.requirements),
            benefits=parsed_response.get("benefits", request.benefits),
            company_intro=parsed_response.get("company_intro", ""),
            response_time=0,  # 임시값, 상위에서 실제 시간으로 설정
            model_used="GPT-4o-mini",
            confidence_score=0.8
        )
    
    async def _generate_with_quality_model(
        self, 
        request: JobPostingRequest, 
        context: str
    ) -> JobPostingResponse:
        """고품질 모델로 채용공고 생성"""
        prompt = self._build_prompt(request, context, "quality")
        
        response = await asyncio.to_thread(
            self.openai_client.chat.completions.create,
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=Config.MAX_TOKENS,
            temperature=0.5
        )
        
        content = response.choices[0].message.content
        parsed_response = self._parse_llm_response(content)
        
        return JobPostingResponse(
            title=parsed_response.get("title", f"{request.position} 모집"),
            description=parsed_response.get("description", ""),
            requirements=parsed_response.get("requirements", request.requirements),
            benefits=parsed_response.get("benefits", request.benefits),
            company_intro=parsed_response.get("company_intro", ""),
            response_time=0,  # 임시값, 상위에서 실제 시간으로 설정
            model_used="GPT-4.1",
            confidence_score=0.95
        )
    
    async def _generate_with_specialized_model(
        self, 
        request: JobPostingRequest, 
        context: str
    ) -> JobPostingResponse:
        """전문 모델로 채용공고 생성"""
        prompt = self._build_prompt(request, context, "specialized")
        
        response = await asyncio.to_thread(
            self.anthropic_client.messages.create,
            model="claude-opus-4-1-20250805",
            max_tokens=Config.MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        parsed_response = self._parse_llm_response(content)
        
        return JobPostingResponse(
            title=parsed_response.get("title", f"{request.position} 모집"),
            description=parsed_response.get("description", ""),
            requirements=parsed_response.get("requirements", request.requirements),
            benefits=parsed_response.get("benefits", request.benefits),
            company_intro=parsed_response.get("company_intro", ""),
            response_time=0,  # 임시값, 상위에서 실제 시간으로 설정
            model_used="Claude-4.1",
            confidence_score=0.9
        )
    
    def _build_prompt(
        self, 
        request: JobPostingRequest, 
        context: str, 
        model_type: str
    ) -> str:
        """프롬프트 구성"""
        base_prompt = f"""
당신은 잡코리아의 채용공고 작성 전문가입니다.
다음 규칙을 엄격히 준수해야 합니다:

1. 제공된 기업 정보만을 사용하여 작성
2. 성별, 연령, 외모 등 차별적 표현 금지
3. 민감한 정보(연봉, 개인정보) 포함 금지
4. 부정확한 정보 생성 금지
5. 한국 노동법 및 채용 관련 법규 준수

기업 정보:
- 회사명: {request.company_name}
- 회사 상세 정보: {json.dumps(request.company_info, ensure_ascii=False)}

직무 정보:
- 포지션: {request.position}
- 요구사항: {', '.join(request.requirements)}
- 복리후생: {', '.join(request.benefits)}

참고 컨텍스트:
{context}

위 정보를 바탕으로 전문적이고 매력적인 채용공고를 작성해주세요.
응답은 다음 JSON 형식으로 제공해주세요:

{{
    "title": "채용공고 제목",
    "description": "상세한 직무 설명",
    "requirements": ["요구사항1", "요구사항2"],
    "benefits": ["복리후생1", "복리후생2"],
    "company_intro": "회사 소개"
}}
"""
        return base_prompt
    
    def _parse_llm_response(self, content: str) -> Dict:
        """LLM 응답 파싱"""
        try:
            # JSON 부분 추출
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
        except Exception as e:
            logger.error(f"응답 파싱 오류: {e}")
        
        return {}
    
    def _generate_fallback_response(
        self, 
        request: JobPostingRequest, 
        start_time: float
    ) -> JobPostingResponse:
        """폴백 응답 생성"""
        return JobPostingResponse(
            title=f"{request.position} 모집",
            description="상세한 직무 설명은 문의 바랍니다.",
            requirements=request.requirements,
            benefits=request.benefits,
            company_intro=f"{request.company_name}에서 {request.position}을 모집합니다.",
            response_time=time.time() - start_time,
            model_used="Fallback",
            confidence_score=0.5
        )

# 메인 서비스
class JobPostingService:
    def __init__(self):
        self.cache_manager = CacheManager()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.content_safety_filter = ContentSafetyFilter()
        self.rag_engine = RAGEngine()
        self.llm_service = LLMService()
    
    async def generate_job_posting(
        self, 
        request: JobPostingRequest
    ) -> JobPostingResponse:
        """채용공고 생성 메인 로직"""
        start_time = time.time()
        
        try:
            # 1. 캐시 확인
            cached_response = self.cache_manager.get_cached_response(request)
            if cached_response:
                logger.info("캐시된 응답 반환")
                return cached_response
            
            # 2. 복잡도 분석
            complexity = self.complexity_analyzer.analyze_complexity(request)
            logger.info(f"요청 복잡도: {complexity.value}")
            
            # 3. 모델 선택
            model_type = self._select_model(complexity, request.user_preference)
            logger.info(f"선택된 모델: {model_type.value}")
            
            # 4. RAG로 컨텍스트 검색
            context = self.rag_engine.retrieve_relevant_context(request)
            logger.info(f"검색된 컨텍스트 길이: {len(context)}")
            
            # 5. LLM으로 채용공고 생성 (타임아웃 적용)
            try:
                response = await asyncio.wait_for(
                    self.llm_service.generate_job_posting(request, context, model_type),
                    timeout=Config.RESPONSE_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"LLM 응답 타임아웃 ({Config.RESPONSE_TIMEOUT}초)")
                return self._generate_timeout_response(request, start_time)
            
            # 6. Content Safety 검사
            filtered_description, risk_score = self.content_safety_filter.filter_content(
                response.description
            )
            response.description = filtered_description
            
            # 7. 응답 캐싱
            self.cache_manager.cache_response(request, response)
            
            # 8. 성능 로깅
            total_time = time.time() - start_time
            logger.info(f"총 처리 시간: {total_time:.2f}초, 모델: {response.model_used}")
            
            return response
            
        except Exception as e:
            logger.error(f"채용공고 생성 오류: {e}")
            return self._generate_error_response(request, start_time)
    
    def _select_model(
        self, 
        complexity: ComplexityLevel, 
        user_preference: str
    ) -> ModelType:
        """복잡도와 사용자 선호도에 따른 모델 선택"""
        if user_preference == "fast":
            return ModelType.FAST
        elif user_preference == "quality":
            return ModelType.QUALITY
        elif user_preference == "balanced":
            if complexity == ComplexityLevel.SIMPLE:
                return ModelType.FAST
            elif complexity == ComplexityLevel.COMPLEX:
                return ModelType.QUALITY
            else:
                return ModelType.SPECIALIZED
        # 기본값으로 SPECIALIZED 반환 (사용자 선호도가 예상치 못한 값인 경우)
        return ModelType.SPECIALIZED
    
    def _generate_error_response(
        self, 
        request: JobPostingRequest, 
        start_time: float
    ) -> JobPostingResponse:
        """오류 시 기본 응답 생성"""
        return JobPostingResponse(
            title=f"{request.position} 모집",
            description="일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            requirements=request.requirements,
            benefits=request.benefits,
            company_intro=f"{request.company_name}에서 {request.position}을 모집합니다.",
            response_time=time.time() - start_time,
            model_used="Error",
            confidence_score=0.0
        )
    
    def _generate_timeout_response(
        self, 
        request: JobPostingRequest, 
        start_time: float
    ) -> JobPostingResponse:
        """타임아웃 시 기본 응답 생성"""
        return JobPostingResponse(
            title=f"{request.position} 모집",
            description="응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
            requirements=request.requirements,
            benefits=request.benefits,
            company_intro=f"{request.company_name}에서 {request.position}을 모집합니다.",
            response_time=time.time() - start_time,
            model_used="Timeout",
            confidence_score=0.0
        )

# 예시 사용
async def main():
    """메인 실행 함수"""
    # 서비스 초기화
    service = JobPostingService()
    
    # 테스트 요청 생성
    request = JobPostingRequest(
        company_name="테크스타트업",
        position="LLM Engineer",
        requirements=["Python", "LangChain", "3년 이상 경력"],
        benefits=["원격근무", "유연근무", "성과급"],
        company_info={
            "industry": "IT/소프트웨어",
            "size": "50-100명",
            "location": "경기도 판교"
        },
        user_preference="balanced"
    )
    
    try:
        # 채용공고 생성
        response = await service.generate_job_posting(request)
        
        # 결과 출력
        print("=== 생성된 채용공고 ===")
        print(f"제목: {response.title}")
        print(f"설명: {response.description}")
        print(f"요구사항: {response.requirements}")
        print(f"복리후생: {response.benefits}")
        print(f"회사 소개: {response.company_intro}")
        print(f"응답 시간: {response.response_time:.2f}초")
        print(f"사용 모델: {response.model_used}")
        print(f"신뢰도: {response.confidence_score:.2f}")
        
    except Exception as e:
        logger.error(f"메인 실행 오류: {e}")

if __name__ == "__main__":
    asyncio.run(main())
