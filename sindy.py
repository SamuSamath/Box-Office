import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('mini.csv')  # Ensure the correct path to your dataset

# Load the pre-trained model using pickle
with open('movie_revenue_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App
st.title("Movie Revenue Prediction Dashboard")

# Sidebar input: Select a movie name
st.sidebar.header('Select Movie Name')
movie_name = st.sidebar.selectbox('Movie Name', df['Movie Name'].unique())

# Automatically extract relevant features for the selected movie
selected_movie_data = df[df['Movie Name'] == movie_name].iloc[0]

# Display the auto-filled relevant features
st.sidebar.write(f"**Release Period:** {selected_movie_data['Release Period']}")
st.sidebar.write(f"**Whether Remake:** {'Yes' if selected_movie_data['Whether Remake'] == 1 else 'No'}")
st.sidebar.write(f"**Whether Franchise:** {'Yes' if selected_movie_data['Whether Franchise'] == 1 else 'No'}")
st.sidebar.write(f"**Genre:** {selected_movie_data['Genre']}")
st.sidebar.write(f"**New Actor:** {'Yes' if selected_movie_data['New Actor'] == 1 else 'No'}")
st.sidebar.write(f"**New Director:** {'Yes' if selected_movie_data['New Director'] == 1 else 'No'}")
st.sidebar.write(f"**New Music Director:** {'Yes' if selected_movie_data['New Music Director'] == 1 else 'No'}")
st.sidebar.write(f"**Lead Star:** {selected_movie_data['Lead Star']}")
st.sidebar.write(f"**Director:** {selected_movie_data['Director']}")
st.sidebar.write(f"**Music Director:** {selected_movie_data['Music Director']}")
st.sidebar.write(f"**Number of Screens:** {selected_movie_data['Number of Screens']}")
st.sidebar.write(f"**Budget (INR):** {selected_movie_data['Budget(INR)']}")
actual_revenue = selected_movie_data['Revenue(INR)']

# Optional: Input for actual revenue
# actual_revenue = st.sidebar.number_input("Actual Revenue (optional)", value=0)

# Predict button
if st.sidebar.button("Predict Revenue"):
    # Collect the movie's feature data in a DataFrame
    user_input = pd.DataFrame({
        'Movie Name': [selected_movie_data['Movie Name']],
        'Release Period': [selected_movie_data['Release Period']],
        'Whether Remake': [selected_movie_data['Whether Remake']],
        'Whether Franchise': [selected_movie_data['Whether Franchise']],
        'Genre': [selected_movie_data['Genre']],
        'New Actor': [selected_movie_data['New Actor']],
        'New Director': [selected_movie_data['New Director']],
        'New Music Director': [selected_movie_data['New Music Director']],
        'Lead Star': [selected_movie_data['Lead Star']],
        'Director': [selected_movie_data['Director']],
        'Music Director': [selected_movie_data['Music Director']],
        'Number of Screens': [selected_movie_data['Number of Screens']],
        'Budget(INR)': [selected_movie_data['Budget(INR)']]
    })

    # Initialize LabelEncoders for categorical features
    le_movie = LabelEncoder()
    le_release_period = LabelEncoder()
    le_wr = LabelEncoder()
    le_wf = LabelEncoder()
    le_release_period = LabelEncoder()
    le_genre = LabelEncoder()
    le_lead_star = LabelEncoder()
    le_director = LabelEncoder()
    le_music_director = LabelEncoder()
    le_na = LabelEncoder()
    le_nd = LabelEncoder()
    le_nmd = LabelEncoder()

    # Fit LabelEncoders to respective columns
    user_input['Movie Name'] = le_movie.fit_transform(user_input['Movie Name'])
    user_input['Release Period'] = le_release_period.fit_transform(user_input['Release Period'])
    user_input['Whether Remake'] = le_wr.fit_transform(user_input['Whether Remake'])
    user_input['Whether Franchise'] = le_wf.fit_transform(user_input['Whether Franchise'])
    user_input['Genre'] = le_genre.fit_transform(user_input['Genre'])
    user_input['Lead Star'] = le_lead_star.fit_transform(user_input['Lead Star'])
    user_input['Director'] = le_director.fit_transform(user_input['Director'])
    user_input['Music Director'] = le_music_director.fit_transform(user_input['Music Director'])
    user_input['New Actor'] = le_na.fit_transform(user_input['New Actor'])
    user_input['New Director'] = le_nd.fit_transform(user_input['New Director'])
    user_input['New Music Director'] = le_nmd.fit_transform(user_input['New Music Director'])
# le = LabelEncoder()
# for col in [user_input['Movie Name'], user_input['Release Period'],user_input['Whether Remake'],user_input['Whether Franchise'],user_input['Genre'],user_input['Lead Star'],user_input['Director'],user_input['Music Director'],user_input['New Actor'],user_input['New Director'],user_input['New Music Director']]:
#     df[col] = le.fit_transform(df[col])
    # Make prediction
    predicted_revenue = model.predict(user_input)[0]

    # Display results
    st.subheader(f"Predicted Revenue: INR {predicted_revenue:,.2f}")

    # If actual revenue is provided, compare with predicted revenue
    if actual_revenue > 0:
        st.subheader(f"Actual Revenue: INR {actual_revenue:,.2f}")

        # Show a chart comparing actual vs predicted
        data = pd.DataFrame({
            'Type': ['Actual', 'Predicted'],
            'Revenue (INR)': [actual_revenue, predicted_revenue]
        })

        fig, ax = plt.subplots()
        ax.bar(data['Type'], data['Revenue (INR)'], color=['blue', 'orange'])
        ax.set_ylabel("Revenue (INR)")
        st.pyplot(fig)
