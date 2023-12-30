
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

bc = load_breast_cancer()
df = pd.DataFrame(data=bc.data, columns=bc.feature_names)
df["target"] = bc.target

malignant_subset_df = df.loc[df["target"] == 0].sample(30)

new_df = pd.concat([malignant_subset_df, df.loc[df["target"] == 1]])
new_df = new_df.sample(frac=1).reset_index(drop=True)

new_df.groupby(["target"]).count()[["mean radius"]].rename(columns={"mean radius":"count"})

# counting the Malignant and Benign patient
d = {0:'Malignant', 1:'Benign'}
countdf = new_df.value_counts(['target']).rename(index=d)


y = new_df["target"].values
X = new_df.drop(columns=["target"]).values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

y_pred_base_model = np.ones(len(y_test))

rf_clf = RandomForestClassifier().fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)




import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=[0,1],
        y=[0,1],
        name="TPR = FPR",
        line=dict(color="black", dash="dash")
    )
)

def PlotRocAuc(y_test, y_pred, color, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test,y_pred)
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            name=f"{model_name}(AUC={auc_score})",
            marker=dict(color=color)
        )
    )
    
PlotRocAuc(y_test, y_pred_rf, "green", "rf_clf")
PlotRocAuc(y_test, y_pred_base_model, "red", "base_clf")

fig.update_layout(title="ROC curve",
                  xaxis_title="False Positive Rate",
                  yaxis_title="True Positive Rate")

#fig.show()


# streamlit 

import streamlit as st

st.title('Breast Cancer Performace classificaiton')
st.write('The dataset presented is named breast_cancer from Sklearn.')

st.dataframe(df)
 
# table counter of the Malignant and Benign patient
st.write('The table below shows the number of Malignent and Benign patient. For this dataset, we are going to exclude some Malignant patient for a more accurate prediction due to its rarity in the real world.')
st.write('Counted from `target` :  0 = Malignant : 1 = Benign ')
st.dataframe(countdf)



st.header('A comparison between two models')
st.write('To find the accuracy/percision of Malignant patient with Breast_Cancer data, we are going to compare 2 accuracy models. These models are the Base Model Classifier and the Random Forest Classifier.')



st.subheader("Accuracy")
st.write("base model accuracy: ", {accuracy_score(y_test, y_pred_base_model)})
st.write("RF model accuracy: ", {accuracy_score(y_test, y_pred_rf)})

st.subheader("Base classifier (base_clf)")
st.write("Precision Score: " , precision_score(y_test, y_pred_base_model))
st.write("Recall Score: " , recall_score(y_test, y_pred_base_model))

st.subheader("Random forest classification (rf_clf)")
st.write("Precision Score: " , precision_score(y_test, y_pred_rf))
st.write("Recall Score: " , recall_score(y_test, y_pred_rf))

st.subheader('ROC curve and AUC score')
st.write('An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds')
st.write('AUC stands for "Area under the ROC Curve." In other words, the AUC calculates the whole two-dimensional area from (0,0) to (1,1) beneath the whole ROC curve')
st.plotly_chart(fig)

st.write('As we can see from the figure, the Rf model has a higher ROC score than the Base model. From this comparison we can conclude that the Random Forest Classifier have the best accuracy model')