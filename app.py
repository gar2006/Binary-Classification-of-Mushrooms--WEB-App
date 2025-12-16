import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score
)

def main():
    st.sidebar.title("Binary Classification Web App")
     
    st.sidebar.markdown("Are your mushrooms edible or poisonous?🍄")

    @st.cache_data(persist=True)
    def load_data():
        df=pd.read_csv(r"C:\Users\garim\Downloads\mushrooms.csv")
        label=LabelEncoder()
        for col in df.columns:
            df[col]=label.fit_transform(df[col])
        return df
    
    @st.cache_data(persist=True)
    def split(df):
        y=df['class']
        X=df.drop(columns=['class'])
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
        return X_train,X_test,y_train,y_test
    
    def plot_metrics(metrics_list):
        #fig, ax = plt.subplots()
        #ax.scatter([1, 2, 3], [1, 2, 3])
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            fig,ax= plt.subplots()
            ConfusionMatrixDisplay.from_estimator(
                model, X_test,y_test, display_labels=['Edible','Poisonous'],ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            fig,ax=plt.subplots()
            RocCurveDisplay.from_estimator(
                model, X_test,y_test,ax=ax)
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            fig,ax=plt.subplots()
            PrecisionRecallDisplay.from_estimator(
                model, X_test,y_test,ax=ax)
            st.pyplot(fig)

    df=load_data()
    class_names=['edible','poisonous']
    if st.sidebar.checkbox("Show Raw Data", False):
        st.subheader("Mushroom Dataset for Classification")
        st.write(df)
        
    X_train,X_test,y_train,y_test=split(df)
    st.sidebar.subheader('Choose Classifier')
    classifier=st.sidebar.selectbox('Classifier', ('Support Vector machine(SVM)','Logistic Regression','Random Forest'))

    if classifier== 'Support Vector machine(SVM)':
        st.sidebar.subheader('Model Hyperparamters')
        C=st.sidebar.number_input("C (Regularisation paramteer)", 0.01, 10.0,step=0.01,key='C_SVM')
        kernel=st.sidebar.radio('Kernel',('rbf','linear'),key='kernel')
        gamma=st.sidebar.radio('Gamma (Kernel Coefficient)', ('scale','auto'), key='gamma')
        
        metrics=st.sidebar.multiselect('What metrics to plot?',('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button('Classify',key='Classify'):
            st.subheader('Support Vector Machine(SVM) Results')
            model=SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(X_train,y_train)
            model.score(X_test,y_test)
            y_pred=model.predict(X_test)
            accuracy=accuracy_score(y_test,y_pred)
            st.write("Accuracy",round(accuracy,2))
            st.write('Precision: ',round(precision_score(y_test,y_pred),2))
            st.write('Recall: ',round(recall_score(y_test,y_pred),2))
            plot_metrics(metrics)
    if classifier== 'Logistic Regression':
        st.sidebar.subheader('Model Hyperparamters')
        C=st.sidebar.number_input("C (Regularisation paramteer)", 0.01, 10.0,step=0.01,key='C_LR')
        max_iter=st.sidebar.slider("Maximum Number of Iterations", 100,500,key='max_iter')
        
        
        metrics=st.sidebar.multiselect('What metrics to plot?',('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button('Classify',key='Classify'):
            st.subheader('Logistic Regression Results')
            model=LogisticRegression(C=C,max_iter=max_iter)
            model.fit(X_train,y_train)
            model.score(X_test,y_test)
            y_pred=model.predict(X_test)
            accuracy=accuracy_score(y_test,y_pred)
            st.write("Accuracy",round(accuracy,2))
            st.write('Precision: ',round(precision_score(y_test,y_pred),2))
            st.write('Recall: ',round(recall_score(y_test,y_pred),2))
            plot_metrics(metrics)
    if classifier== 'Random Forest':
        st.sidebar.subheader('Model Hyperparamters')
        n_estimators=st.sidebar.number_input("The number of trees in the forest", 100,5000,step=10,key='n_estimators')
        bootstrap=st.sidebar.radio('Bootstrap samples when building trees',(True,False), key='bootstrap')
        max_depth=st.sidebar.number_input('The maximum depth of the tree', 1,20,step=1,key='max_depth')
        metrics=st.sidebar.multiselect('What metrics to plot?',('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button('Classify',key='Classify'):
            st.subheader('Rnadom Forest Results')
            model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
    
            model.fit(X_train,y_train)
            model.score(X_test,y_test)
            y_pred=model.predict(X_test)
            accuracy=accuracy_score(y_test,y_pred)
            st.write("Accuracy",round(accuracy,2))
            st.write('Precision: ',round(precision_score(y_test,y_pred),2))
            st.write('Recall: ',round(recall_score(y_test,y_pred),2))
            plot_metrics(metrics)
            

if __name__ == "__main__":
    main()
