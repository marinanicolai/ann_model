# 📦 1. Import Libraries
import streamlit as st
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O

import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

st.title(" ANN Model Explorer by Marina")

#kaggle link to data: https://www.kaggle.com/datasets/shivan118/churn-modeling-dataset/data

