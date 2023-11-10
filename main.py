import streamlit as st
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain

# 스트림릿 인터페이스
st.title("LangChain LlamaCPP 테스트")

# 모델 경로 설정
model_path = "mistral-7b-openorca.Q4_0.gguf"

# 프롬프트 템플릿 설정
template = """Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

# 콜백 매니저 설정
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# LlamaCPP 모델 초기화 (CPU)
llm = LlamaCpp(
    model_path=model_path, 
    callback_manager=callback_manager, 
    verbose=True, 
    temperature=0
)

# LLMChain 설정
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 스트림릿 사용자 입력
user_input = st.text_area("질문을 입력하세요:", value="", max_chars=500)

# 실행 버튼
if st.button("답변 생성"):
    # 사용자 질문에 대한 답변 실행
    answer = llm_chain.run(user_input)
    st.write(answer)
