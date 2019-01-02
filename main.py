import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("profiles.csv")

df.fillna('no_answer', inplace=True)

# # the question is how well can a persons diet and substance use determine age income or body_type

# income_by_age = df.groupby(df.age).income.mean()
# income_by_age.plot.bar(rot=0)
# plt.xlabel("Age")
# plt.ylabel("Mean Income")
# plt.show()

# income_by_body_type = df.groupby(df.body_type).income.mean()
# income_by_body_type.plot.bar(rot=0)
# plt.xlabel("Body Type")
# plt.ylabel("Mean Income")
# plt.show()

def percentage_body_type_by_other(other_column):
    results = {}
    for f in df.groupby(df[other_column]): 
        results[f[0]] = f[1].body_type.value_counts()

    two_way = pd.DataFrame()
    for label, value_count in results.items():
        value_count.name = label
        value_count = (value_count / value_count.sum()) * 100
        row = value_count.to_frame().transpose()
        two_way = pd.concat([two_way, row], axis=0)

    two_way.plot.bar(rot=0)
    plt.title("Percentage " + other_column + " by Body Type")
    plt.xlabel(other_column + " status")
    plt.xticks(rotation=90)
    plt.ylabel("Percentage responders Body Type")
    plt.show()

# # for label in ["drinks", "drugs", "smokes", "diet"]:
# #     percentage_body_type_by_other(label)

# import pdb; pdb.set_trace()

body_type_map = {
    "no_answer": 0,
    "rather not say": 1,
    "used up": 2,
    "overweight": 3, 
    "curvy": 4,
    "full figured": 5, 
    "a little extra": 6,
    "skinny": 7,
    "thin": 8, 
    "average": 9,
    "fit": 10,
    "athletic": 11,
    "jacked": 12
}

drug_type_map = {
    "no_answer": 0,
    "never": 1,
    "sometimes": 2,
    "often": 3
}

drink_type_map = {
    "no_answer": 0,
    "not at all": 1,
    "rarely": 2,
    "socially": 3,
    "often": 4,
    "very often": 5,
    "desperately": 6
}

smoke_type_map = {
    "no_answer": 0,
    "no": 1,
    "trying to quit": 2,
    "when drinking": 3,
    "sometimes": 4,
    "yes": 5
}

diet_type_map = {
    "no_answer": 0,
    "strictly vegan": 1, 
    "vegan": 2,
    "mostly vegan": 3, 
    "strictly vegetarian": 4, 
    "vegetarian": 5, 
    "mostly vegetarian": 6, 
    "mostly anything": 7,
    "anything": 8,
    "strictly anything": 9
}



feature_data = df[["diet", "drinks", "drugs", "body_type", "smokes", "income", "age"]]

feature_data["diet_code"]  = feature_data.diet.map(diet_type_map)
feature_data["drinks_code"]  = feature_data.drinks.map(drink_type_map)
feature_data["smokes_code"]  = feature_data.smokes.map(smoke_type_map)
feature_data["drugs_code"]  = feature_data.drugs.map(drug_type_map)
feature_data["body_code"]  = feature_data.body_type.map(body_type_map)

feature_data.dropna(inplace=True)
feature_data = feature_data[feature_data.income != -1]


def multiple_linear_regression(feature_data, dependent_column):
    dependent_data = feature_data[dependent_column]
    labels_to_drop = ["diet", "drinks", "drugs", "body_type", "smokes"] + [dependent_column]
    feature_data.drop(labels=labels_to_drop, axis=1, inplace=True)

    feature_values = feature_data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    feature_values_scaled = min_max_scaler.fit_transform(feature_values)

    feature_data = pd.DataFrame(feature_values_scaled, columns=feature_data.columns)

    X_train, X_test, y_train, y_test = train_test_split(feature_data, dependent_data, test_size = 0.2, random_state = 1)

    model = LinearRegression()
    model.fit(X_train,y_train)

    print('Train Score:', model.score(X_train,y_train))
    print('Test Score:', model.score(X_test,y_test))
    print(sorted(list(zip(feature_data.columns,model.coef_)),key = lambda x: abs(x[1]),reverse=True))

    y_predicted = model.predict(X_test)

    plt.scatter(y_test, y_predicted, alpha=0.4)
    plt.xlabel(dependent_column)
    plt.ylabel('Predicted ' + dependent_column)
    plt.show()


# multiple_linear_regression(feature_data, "age")
multiple_linear_regression(feature_data, "body_code")


import pdb; pdb.set_trace()
