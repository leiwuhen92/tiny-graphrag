from openai import OpenAI
from langchain.embeddings.base import Embeddings
from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import requests
from typing import Any, List, Optional, Dict, Mapping
import logging


class GpuStackEmbeddings(Embeddings):
    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

    def embed_documents(self, text: str) -> List[float]:
        """为文档生成嵌入向量"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": text,
            "model": self.model_name
        }
        response = requests.post(f"{self.api_url}/embeddings", json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            logging.info(f" result: {result}")
            return result["data"][0]["embedding"]
        else:
            raise Exception(f"Embedding请求失败: {response.status_code} - {response.text}")

    def embed_query(self, text: str) -> List[float]:
        """为查询文本生成嵌入向量"""
        return self.embed_documents([text])[0]


class GPUStackLLM(LLM):
    """符合 LangChain Runnable 接口的 GPUStack LLM 包装器"""
    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return "gpustack"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """调用 LLM 并返回结果"""
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个有帮助的AI助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling GPUStack LLM: {str(e)}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """返回识别参数"""
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
