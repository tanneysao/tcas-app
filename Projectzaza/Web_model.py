import streamlit as st
import pandas as pd
import pickle 




st.image('logo.png', width = 500)
   
st.title('MFU prediction **status**')


st.sidebar.header('User Input') 
st.sidebar.subheader('Please enter your data:')


def get_input():
    
    V_AcademicYear = st.sidebar.selectbox('AcademicYear', [2562,2563])
    v_Sex = st.sidebar.radio('Sex', ['Male','Female'])
    V_FacultyID = st.sidebar.slider('FacultyID', 10, 25, 1)
    V_DepartmentCode = st.sidebar.selectbox('DepartmentCode',[1005, 1006, 1102, 1105, 1112, 1201, 1202, 1203, 1205, 1207, 1209,
       1210, 1301, 1302, 1305, 1306, 1401, 1407, 1501, 1601, 1701, 1703,
       1804, 1806, 1807, 1808, 1901, 2101, 2201, 2301, 2401, 2402, 2403,
       2404, 2501, 2502, 2503])
    V_EntryTypeID = st.sidebar.selectbox('EntryTypeID',[40, 10, 29, 24, 20, 30, 11, 52, 41, 51, 66, 50, 67, 59, 15, 64, 68,
       58, 69])
    V_EntryGroupID = st.sidebar.selectbox('EntryGroupID',[623210, 623020, 623011, 623050, 623030, 623060, 623501, 623043,
       623040, 623041, 623044, 623111, 623110, 623010, 623112, 623055,
       623042, 623080, 623070, 623120, 623071, 623212, 623130, 623141,
       623140, 623182, 623185, 623181, 623184, 623180, 623186, 623190,
       623301, 623302, 623303, 623305, 623306, 623307, 623230, 623401,
       633210, 633020, 633011, 633014, 633030, 633060, 633501, 633051,
       633071, 633041, 633046, 633010, 633601, 633040, 633044, 633212,
       633042, 633050, 633043, 633070, 633012, 633013, 633045, 633401,
       633080, 633190, 633302, 633301, 633303, 633502, 633304, 633305,
       633503])
    V_Tcas = st.sidebar.slider('TCAS', 1, 5, 1)
   
    V_HomeRegion = st.sidebar.selectbox('HomeRegion',['International', 'North', 'North East', 'South', 'Central', 'East',
       'Bankok', 'West'])
    V_StudentTH = st.sidebar.selectbox('StudentTH',[0,1])
    V_Country = st.sidebar.selectbox('Country',['China', 'Korea', 'Thailand',
       'United Kingdom of Great Britain and Northern Ireland',
       'United States of America', 'Laos', 'Philippines', 'Indonesia',
       'South Korea', 'Myanmar', 'Japan', 'Bangladesh', 'South Africa',
       'Bhutan', 'Cameroon', 'Brazil', 'France', 'Taiwan', 'Mali',
       'Australia'])
    V_SchoolRegionNameEng = st.sidebar.selectbox('SchoolRegionNameEng',['Foreign', 'Northern', 'Northeast', 'Southern', 'Central',
       'Eastern', 'Western'])
    V_ReligionName = st.sidebar.selectbox('ReligionName',['พุทธ', 'คริสต์', 'อิสลาม', '-', 'บาไฮ', 'ซิกข์', 'ฮินดู'])
    V_EntryGPA= st.sidebar.number_input('EntryGPA')
    V_Gpax = st.sidebar.number_input('GPAX')
    V_Gpa_eng = st.sidebar.number_input('GPA_Eng')
    V_Gpa_math = st.sidebar.number_input('GPA_Math')
    V_Gpa_sci = st.sidebar.number_input('GPA_Sci')
    V_Gpa_sco = st.sidebar.number_input('GPA_Sco')

  
   

    data = {'Sex': v_Sex,
            'AcademicYear': V_AcademicYear,
            'FacultyID': V_FacultyID,
            'DepartmentCode': V_DepartmentCode,
            'EntryTypeID': V_EntryTypeID,
            'EntryGroupID': V_EntryGroupID,
            'TCAS': V_Tcas,
            'StudentTH': V_StudentTH,
            'Country': V_Country,
            'SchoolRegionNameEng': V_SchoolRegionNameEng,
            'ReligionName': V_ReligionName,
            'HomeRegion': V_HomeRegion,         
            'EntryGPA': V_EntryGPA,     
            'GPAX': V_Gpax,
            'GPA_Eng': V_Gpa_eng,
            'GPA_Math': V_Gpa_math,
            'GPA_Sci': V_Gpa_sci,
            'GPA_Sco': V_Gpa_sco,   
            }


   

    data_df = pd.DataFrame(data, index=[0])
    return data_df


df = get_input()




st.header('It simulators Student\'s :')


st.subheader('User Input:')
st.write(df)


data_sample = pd.read_csv('datav3.csv')
df = pd.concat([df, data_sample],axis=0)




cat_data = pd.get_dummies(df[['Sex','Country','SchoolRegionNameEng','ReligionName','HomeRegion']])

X_new = pd.concat([cat_data, df], axis=1)

X_new = X_new[:1] 

X_new = X_new.drop(columns=['Sex','Country','SchoolRegionNameEng','ReligionName','HomeRegion'])


st.subheader('Pre-Processed Input:')
st.write(X_new)


load_sc = pickle.load(open('normalization.pkl', 'rb'))
X_new = load_sc.transform(X_new)

st.subheader('Normalized Input:')
st.write(X_new)


load_knn = pickle.load(open('best_knn.pkl', 'rb'))

prediction = load_knn.predict(X_new)


st.subheader('Prediction:')

st.write(prediction)