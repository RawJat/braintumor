import os
import numpy as np
import tensorflow as tf
import spektral
from spektral.layers import GCNConv
from spektral.data import Graph, Dataset
from spektral.data.loaders import SingleLoader
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split