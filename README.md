# RateMe: The Ultimate App Rating Platform

<a href="http://ec2-3-144-149-146.us-east-2.compute.amazonaws.com:8080/">
<p align="center">
<img src="https://github.com/NatsuD99/Google_Play_Store/assets/58219175/906c18d8-8286-4b20-8500-7d2a59958d39" width="400" height="340" title="RateMe" alt="RateMe Logo">
</p>
</a>

Welcome to RateMe, your go-to platform for estimating the potential rating of your Android app before even hitting the market. Whether you're a seasoned developer looking to optimize your app's features or a novice seeking guidance for your first app, RateMe has you covered. This platform leverages a sophisticated  Light Gradient Boosting Machine (LGBM) model, built on 14 distinct features and over 2 Million data, to predict the potential success of your Android application with an error margin of 0.19(RMSE) on a rating scale of 0-5.

## Features
**Core Attributes**
1. `Categories:` Classify your app into relevant categories, providing a foundational understanding of its purpose.
2. `Ad Supported:` Decide whether your app will include advertisements.
3. `Minimum Android Version:` Specify the minimum Android version required for optimal performance.
4. `In App Purchases:` Indicate if your app will offer in-app purchases to enhance monetization.
5. `Content Rating:` Define the appropriate content rating for your app's target audience.
6. `Has Developer Website:` Enhance credibility by providing a link to your developer website.
7. `Free:` Indicate whether your app will be free to download and use.
8. `Has Privacy Policy:` Include a privacy policy for transparency and user trust.

**Dynamic Modifiers**
1. `Rating Count:` Understand the popularity of similar apps by examining their rating counts.
2. `Maximum Installs:` Set an upper limit for the number of installs, guiding your app's scalability.
3. `Minimum Installs:` Specify the lower boundary for the number of installs your app is targeting.
4. `App Age:` Experiment with the age of your app to gauge its impact on user engagement.
5. `Size:` Adjust the size of your app to optimize user experience and device compatibility.
6. `Price:` If your app is not free, set the price to influence user perceptions.

## Usage
1. **Developers:**

    * Input the core attributes and dynamic modifiers of your app.
    * Experiment with dynamic modifiers to fine-tune your app's potential rating.
    * Utilize the predicted rating to optimize your app's features for better user reception.

2. **Aspiring Developers:**

    * Leverage RateMe to gain insights into the potential success of your app concept.
    * Experiment with different combinations of features to estimate the impact on the predicted rating.

## Model Details
Our machine learning model, powered by Light Gradient Boosting Machine (LGBM), boasts an impressive Root Mean Square Error (RMSE) of 0.19. The rating scale is from 0 to 5, providing a precise estimation of your app's potential user rating. The model is trained on the 14 carefully selected features mentioned above, ensuring a comprehensive analysis of your app's attributes.

## Deployment
The RateMe platform is deployed as a user-friendly website. Leveraging the power of Flask and the accessibility of Heroku, our platform offers a user-friendly and responsive interface, enabling users to effortlessly navigate through input forms and receive accurate predictions for their app's potential rating.

## Contributions
If you want to run this project and make changes on your own, keep the following in mind.
1. Download your kaggle credentials and put it in the folder you are working on, in order to fetch the data in the code.
2. Run model.py to generate the pickled model.
3. run app.py to run the flask model locally.

We welcome contributions from the community to enhance and improve the RateMe platform. Whether you're a developer, data scientist, or enthusiast, your input is valuable.

