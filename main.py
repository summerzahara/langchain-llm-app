import langchain_helper as lch
import streamlit as st

st.title('Pet Name Generator')

user_animal_type = st.sidebar.selectbox(
    'What is your pet?',
    ('Cat', 'Dog', 'Parrot', 'Snake', 'Chicken')
)

user_pet_color = st.sidebar.text_area(
    f'What color is your {user_animal_type.lower()}?',
    max_chars=15
)

if user_pet_color:
    response = lch.gen_pet_name(user_animal_type, user_pet_color)
    st.text(response['pet_name'])