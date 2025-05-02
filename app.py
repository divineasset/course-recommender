import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load data and model
df = pd.read_csv("personalized_learning_dataset.csv")
model = tf.keras.models.load_model("recommender_model.h5")

# Normalize and encode like before
engagement_features = [
    'Time_Spent_on_Videos', 'Quiz_Scores',
    'Assignment_Completion_Rate', 'Final_Exam_Score', 'Feedback_Score'
]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[engagement_features] = scaler.fit_transform(df[engagement_features])
df['interaction_score'] = df[engagement_features].mean(axis=1)

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df['user_id'] = user_encoder.fit_transform(df['Student_ID'])
df['item_id'] = item_encoder.fit_transform(df['Course_Name'])

# Recommendation function
def recommend_courses(student_id, model, df, user_encoder, item_encoder, top_n=5):
    if student_id not in df['Student_ID'].values:
        return ["Student ID not found."]
    
    user_int_id = user_encoder.transform([student_id])[0]
    all_items = df['item_id'].unique()
    user_ids = np.full_like(all_items, user_int_id)

    scores = model.predict([user_ids, all_items], verbose=0).flatten()
    top_indices = np.argsort(scores)[-top_n:][::-1]
    top_item_ids = all_items[top_indices]
    recommended_courses = item_encoder.inverse_transform(top_item_ids)

    return list(recommended_courses)

# Streamlit UI
st.title("🎓 Personalized Learning Recommender System")

st.markdown("Select a student to view personalized course suggestions based on their engagement and performance.")

student_list = df['Student_ID'].unique()
selected_student = st.selectbox("Choose Student ID", student_list)

top_n = st.slider("Number of courses to recommend", 1, 10, 5)

if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        recommendations = recommend_courses(selected_student, model, df, user_encoder, item_encoder, top_n=top_n)
        st.success(f"Top {top_n} recommended courses for **{selected_student}**:")
        for i, course in enumerate(recommendations, 1):
            st.write(f"{i}. {course}")
