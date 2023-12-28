from flask import Flask, request, render_template
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)
model = pickle.load(open('rating_model.pkl', 'rb'))
categories = [
    "Action", "Adventure", "Arcade", "Art & Design", "Auto & Vehicles",
    "Beauty", "Board", "Books & Reference", "Business", "Card",
    "Casino", "Casual", "Comics", "Communication", "Dating",
    "Education", "Entertainment", "Events", "Finance", "Food & Drink",
    "Health & Fitness", "House & Home", "Libraries & Demo", "Lifestyle",
    "Maps & Navigation", "Medical", "Music", "News & Magazines",
    "Parenting", "Personalization", "Photography", "Productivity",
    "Puzzle", "Racing", "Role Playing", "Shopping", "Simulation",
    "Social", "Sports", "Strategy", "Tools", "Travel & Local",
    "Trivia", "Video Players & Editors", "Weather", "Word"
]

# %%

@app.route('/')
def home():
    return render_template('index.html', categories=categories)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        features = {
            "Category": [request.form['Category']],  # text
            "Rating Count": [request.form['Rating Count']],  # number
            "Maximum Installs": [request.form['Maximum Installs']],  # number
            "App Age": [request.form['App Age']],  # number
            "Size": [request.form['Size']],  # number
            "Ad Supported": [request.form['Ad Supported']],  # select
            "Minimum Android": [request.form['Minimum Android']],  # number
            "In App Purchases": [request.form['In App Purchases']],  # select
            "Content Rating": [request.form['Content Rating']],  # number
            "Has Developer Website": [request.form['Has Developer Website']],  # select
            "Price": [request.form['Price']],  # number
            "Free": [request.form['Free']],  # select
            "Has Privacy Policy": [request.form['Has Privacy Policy']],  # select
            "Minimum Installs": [request.form['Minimum Installs']]  # number
        }


        features = pd.DataFrame(features)
        label_mapping = {
            'Action': 0, 'Adventure': 1, 'Arcade': 2, 'Art & Design': 3, 'Auto & Vehicles': 4,
            'Beauty': 5, 'Board': 6, 'Books & Reference': 7, 'Business': 8, 'Card': 9,
            'Casino': 10, 'Casual': 11, 'Comics': 12, 'Communication': 13, 'Dating': 14,
            'Education': 15, 'Entertainment': 16, 'Events': 17, 'Finance': 18, 'Food & Drink': 19,
            'Health & Fitness': 20, 'House & Home': 21, 'Libraries & Demo': 22, 'Lifestyle': 23,
            'Maps & Navigation': 24, 'Medical': 25, 'Music': 26, 'News & Magazines': 27,
            'Parenting': 28, 'Personalization': 29, 'Photography': 30, 'Productivity': 31,
            'Puzzle': 32, 'Racing': 33, 'Role Playing': 34, 'Shopping': 35, 'Simulation': 36,
            'Social': 37, 'Sports': 38, 'Strategy': 39, 'Tools': 40, 'Travel & Local': 41,
            'Trivia': 42, 'Video Players & Editors': 43, 'Weather': 44, 'Word': 45
        }

        features['Category'] = features['Category'].map(label_mapping).astype('int32')
        features['Rating Count'] = features['Rating Count'].astype('float64')
        features['Minimum Installs'] = features['Minimum Installs'].astype('float64')
        features['Maximum Installs'] = features['Maximum Installs'].astype('float64')
        features['Free'] = features['Free'].astype('bool')
        features['Price'] = features['Price'].astype('float64')
        features['Size'] = features['Size'].astype('float64')
        features['Minimum Android'] = features['Minimum Android'].astype('int32')

        content_rating_mapping = {'10+': 0, '17+': 1, '18+': 2, 'Everyone': 3, 'Teen': 4, 'Unrated': 5}
        features['Content Rating'] = features['Content Rating'].map(content_rating_mapping).astype('int32')

        features['Ad Supported'] = features['Ad Supported'].replace({'No': 0, 'Yes': 1}).astype('int64')
        features['In App Purchases'] = features['In App Purchases'].replace({'No': 0, 'Yes': 1}).astype('int64')
        features['App Age'] = features['App Age'].astype('float64')
        features['Has Privacy Policy'] = features['Has Privacy Policy'].replace({'No': 0, 'Yes': 1}).astype('int64')
        features['Has Developer Website'] = features['Has Developer Website'].replace({'No': 0, 'Yes': 1}).astype(
            'int32')
        prediction = model.predict(features)
        output = round(prediction[0], 1)

        # return features.head(), features.info()
        return render_template('index.html', prediction_text='Estimated App Rating: {}'.format(output))
        # return {'Prediction': output}
    return render_template('index.html')
# %%
# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)