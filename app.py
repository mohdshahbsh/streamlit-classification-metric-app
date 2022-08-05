import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit import session_state as ss

def main():
    st.title("Classification Metric Comparison")
    st.sidebar.subheader("Adjust Parameter")

    # input number of samples
    st.sidebar.number_input("Number of samples",10,100000,value=100,key="sample")

    # button for balance ratio
    balance_ratio = np.arange(0.1,0.6,0.1).round(1)
    st.sidebar.radio("Balance Ratio", options=balance_ratio, index=4, key='ratio')
    
    # calculate tp and tn
    total_tp = np.int(ss.ratio * ss.sample)
    total_tn = np.int(ss.sample - total_tp)  
    
    # slider for changing the TP/FP/TN/FN
    tp = st.sidebar.slider("True Positive", 0, total_tp, value=total_tp//2, key='tp')
    tn = st.sidebar.slider("True Negative", 0, total_tn, value=total_tn//2, key='tn')
    fn = abs(total_tp - tp)
    fp = abs(total_tn - tn)

    # CONFUSION MATRIX
    left_col, right_col = st.columns(2)
    ## to seperate charts by columns use st.columns
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(data=np.array([[tn,fp],[fn,tp]]), annot=True)
    ## label confusion matrix
    ax_cm.set_xlabel("Predicted Value")
    ax_cm.set_ylabel("True Label")
    left_col.pyplot(fig_cm)

    # PERFORMANCE METRIC
    right_col.markdown(f"$Accuracy = {(tp+tn)/(tp+tn+fp+fn):0.1%}$%")
    right_col.markdown(f"$Balanced Accuracy = {((tp/(tp+fn))+(tn/(tn+fp)))/2:0.1%}$%")    
    right_col.markdown(f"$F1-Score = {(tp/(tp+0.5*(fp+fn))):0.1%}$%")
    right_col.markdown(f"$Sensitivity = {tp/(tp+fn):0.1%}$%")
    right_col.markdown(f"$Specificity\,= {tn/(tn+fp):0.1%}$%")
    right_col.markdown(f"$Precision_1\,= {tp/(tp+fp):0.1%}$%") 
    right_col.markdown(f"$Precision_0\,= {tn/(tn+fn):0.1%}$%")
       

if __name__ == "__main__":
    main()