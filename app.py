import streamlit as st
from transformers import pipeline

# Инициализация моделей с использованием PyTorch
qa_pipeline = pipeline("question-answering", model="DeepPavlov/rubert-base-cased", from_pt=True)
text_gen_pipeline = pipeline("text-generation", model="sberbank-ai/rugpt3small_based_on_gpt2")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")


def chatbot_response(user_input, task):
    if task == 'question':
        response = qa_pipeline(question=user_input['question'], context=user_input['context'])
        return response['answer']
    elif task == 'generate':
        response = text_gen_pipeline(user_input, max_length=50, num_return_sequences=1)
        return response[0]['generated_text']
    elif task == 'summarize':
        response = summarization_pipeline(user_input, max_length=100, min_length=30, do_sample=False)
        return response[0]['summary_text']
    else:
        return "Неизвестная задача."


# Streamlit UI
st.title("Chatbot with Transformers")

task = st.selectbox("Выберите задачу:", ['question', 'generate', 'summarize'])

if task == 'question':
    question = st.text_input("Введите ваш вопрос:")
    context = st.text_area("Введите контекст:")
    if st.button("Получить ответ"):
        if question and context:
            user_input = {'question': question, 'context': context}
            answer = chatbot_response(user_input, task)
            st.write("Ответ:", answer)
        else:
            st.warning("Пожалуйста, заполните все поля.")

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