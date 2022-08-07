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
    fig_tp = trellis_plot(tp,tn,fp,fn)
    right_col.pyplot(fig_tp)
    
    metric_df = pd.DataFrame({
        "Metric" : [ 
            "Accuracy",
            "Balanced Accuracy",
            "Jaccard Index",
            "F1-Score",
            "Recall_0",
            "Recall_1",
            "Precision_0",
            "Precision_1",
        ],
        "Score" : [
            (tp+tn)/(tp+tn+fp+fn), # Accuracy
            ((tp/(tp+fn))+(tn/(tn+fp)))/2, # Bal.Acc
            (tp/(tp+fp+fn)), # Jaccard
            (tp/(tp+0.5*(fp+fn))), # F1
            tn/(tn+fp), # recall 0
            tp/(tp+fn), # recall 1
            (tn/(tn+fn)), # precision 0
            tp/(tp+fp), # precision 1
        ]
    })
    st.table(metric_df)

def trellis_plot(tp,tn,fp,fn):
    # Trellis Plot for Classification Metric
    fig_tp, ax = plt.subplots(figsize=(8,9))
    
    y1_axis = [
        "Accuracy",
        "Balanced Accuracy",
        "Jaccard Index",
        "F1-Score",
        "Recall",
        "Precision"
    ]
    y0_axis = ["Precision","Recall"]
    x0_axis = [
        (tn/(tn+fn)),
        tn/(tn+fp)
    ]
    x1_axis = [
        (tp+tn)/(tp+tn+fp+fn),
        ((tp/(tp+fn))+(tn/(tn+fp)))/2,
        (tp/(tp+fp+fn)),
        (tp/(tp+0.5*(fp+fn))),
        tp/(tp+fn),
        tp/(tp+fp)
    ]

    ## label 1 plot
    ax.plot(x1_axis, y1_axis, 'o', 
        markersize=14,
        color = 'white',
        markeredgecolor = "steelblue",
        markeredgewidth = 4,
    )

    ## label 0 plot
    ax.plot(x0_axis,y0_axis, 'o',
        markersize=14,
        color = 'white',
        markeredgecolor = "coral",
        markeredgewidth = 4,
       )

    plt.title("Performance Metric", fontsize=18, pad=12, loc="left",)
    sns.despine()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0,1.1)
    plt.grid(axis='y', alpha=0.5)
    ## configure legend for the label markers
    plt.legend(
        labels = ['Label 1','Label 0'],
        loc = 'upper right',
        bbox_to_anchor = (1.2,1.0),
        borderpad = 1.2,
        labelspacing = 1.2
    )

    return fig_tp      

if __name__ == "__main__":
    main()