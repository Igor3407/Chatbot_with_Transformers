import streamlit as st
from transformers import pipeline

# Загрузка моделей
qa_pipeline = pipeline("question-answering", model="KirrAno93/rubert-base-cased-finetuned-squad")
text_gen_pipeline = pipeline("text-generation", model="sberbank-ai/rugpt3small_based_on_gpt2")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Фиксированный контекст
fixed_context = """Ваш фиксированный текст для контекста здесь. Вы можете заменить его на любой текст, который нужен."""

def chatbot_response(user_input, task):
    try:
        if task == 'question':
            response = qa_pipeline(question=user_input, context=fixed_context)
            return response['answer']
        elif task == 'generate':
            response = text_gen_pipeline(user_input, max_length=50, num_return_sequences=1)
            return response[0]['generated_text']
        elif task == 'summarize':
            response = summarization_pipeline(user_input, max_length=100, min_length=30, do_sample=False)
            return response[0]['summary_text']
        else:
            return "Неизвестная задача."
    except Exception as e:
        return f"Произошла ошибка: {str(e)}"

# Streamlit UI
st.title("Chatbot with Transformers")

task = st.selectbox("Выберите задачу:", ['question', 'generate', 'summarize'])

if task == 'question':
    question = st.text_input("Введите ваш вопрос:")
    if st.button("Получить ответ"):
        if question:
            answer = chatbot_response(question, task)
            st.write("Ответ:", answer)
        else:
            st.warning("Пожалуйста, введите вопрос.")

elif task == 'generate':
    prompt = st.text_input("Введите текст для генерации:")
    if st.button("Сгенерировать текст"):
        if prompt:
            generated_text = chatbot_response(prompt, task)
            st.write("Сгенерированный текст:", generated_text)
        else:
            st.warning("Пожалуйста, введите текст.")

elif task == 'summarize':
    text_to_summarize = st.text_area("Введите текст для краткого изложения:")
    if st.button("Сделать краткое изложение"):
        if text_to_summarize:
            summary = chatbot_response(text_to_summarize, task)
            st.write("Краткое изложение:", summary)
        else:
            st.warning("Пожалуйста, введите текст.")