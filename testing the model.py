#importing necessary libraries
import pandas as pd
import numpy as np
#to load Label encoder files,
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
from sklearn.preprocessing import LabelEncoder

#load the saved model
import pickle
filename = 'resale_model.sav'
model = pickle.load(open(filename, 'rb'))


#create a dataframe so as to get user values.
new_df = pd.DataFrame(columns =['vehicleType', 'yearOfRegistration', 'gearbox',
                                'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType',
                                'brand', 'notRepairedDamage'] )

#for the UI building, we'll pass here the input values instead of hardcoded ones
#pass the user input values here instead of the values which I have written,
new_row = {'yearOfRegistration':2011, 'powerPS':190, 'kilometer':125000,
       'monthOfRegistration':5, 'gearbox':'automatic', 'notRepairedDamage':'Yes',
       'model':'beetle', 'brand':'volkswagen', 'fuelType':'petrol',
       'vehicleType':'coupe'}

#append a new row to the dataframe formed by user inputs
new_df = new_df.append(new_row,ignore_index = True)


#label encoding for the values we've given dropdown menu for
labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']

#open label encoding files and do the encoding on the data
mapper = {}
for i in labels:
    mapper[i] = LabelEncoder()
    mapper[i].classes_ = np.load(str('classes'+i+'.npy'))
    tr = mapper[i].fit_transform(new_df[i])
    new_df.loc[:, i + '_labels'] = pd.Series(tr, index=new_df.index)


#added label encoded columns to df
labeled = new_df[ ['yearOfRegistration'
                        ,'powerPS'
                        ,'kilometer'
                        ,'monthOfRegistration'
                        ] 
                    + [x+'_labels' for x in labels]]

#created the required dataframe as labeled where we have
#label encoded values for all

print(labeled.columns) #to check all the columns, also see, Price is removed
print(labeled) #printing the df, it has only one row, whose inputs are given by us for now.

#taking the values from X
X = labeled.values

#making prediction and printing the output
y_prediction = model.predict(X)
print("$ ",y_prediction[0].round(2),sep='')

