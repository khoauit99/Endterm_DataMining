import tkinter as tk
from tkinter import messagebox
from tkinter import *
from PIL import ImageTk, Image
import tkinter.font as tkFont
from tkinter.ttk import Combobox, Progressbar
from tkinter import filedialog, ttk
import os
import numpy as np
import csv
from pandas import DataFrame
import Preprocessing
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import load


list_test_2=[None]*18
list_test_1=["1"]*18
list_test_2[0]="5/10/2013"

temp_str_info1=""
temp_str_info2=""

temp_location=""
temp_WindGustDir=""
temp_WindDir9am=""
temp_WindDir3pm=""
temp_RainToday=""

temp_lg=0.0
temp_rd=0
temp_svm=0
temp_dt=0

counting=0

def showketqua():
    #print(temp_location)
    print(temp_WindDir3pm)
    print(temp_RainToday)
    print(temp_location)
    print(temp_WindGustDir)
    print(temp_WindDir9am)

def sampletest():
    date = list_test_2[0]
    location = temp_location
    min_temp = float(list_test_2[1])
    max_temp = float(list_test_2[2])
    rainfall = float(list_test_2[3])
    evaporation = float(list_test_2[4])
    sunshine = float(list_test_2[5])
    windgustdir = temp_WindGustDir
    windgustspeed = float(list_test_2[6])
    winddir9am = temp_WindDir9am
    winddir3pm = temp_WindDir3pm
    windspeed9am = float(list_test_2[7])
    windspeed3pm = float(list_test_2[8])
    humidity9am = float(list_test_2[9])
    humidity3pm = float(list_test_2[10])
    pressure9am = float(list_test_2[11])
    pressure3pm = float(list_test_2[12])
    cloud9am = float(list_test_2[13])
    cloud3pm = float(list_test_2[14])
    temp9am = float(list_test_2[15])
    temp3pm = float(list_test_2[16])
    raintoday = temp_RainToday
    risk_mm = float(list_test_2[17])
    for i in range (0,17):
        print(list_test_2[i])
    # create a pandas dataframe
    data = {'Date': [date],
            'Location': [location],
            'MinTemp': [min_temp],
            'MaxTemp': [max_temp],
            'Rainfall': [rainfall],
            'Evaporation': [evaporation],
            'Sunshine': [sunshine],
            'WindGustDir': [windgustdir],
            'WindGustSpeed': [windgustspeed],
            'WindDir9am': [winddir9am],
            'WindDir3pm': [winddir3pm],
            'WindSpeed9am': [windspeed9am],
            'WindSpeed3pm': [windspeed3pm],
            'Humidity9am': [humidity9am],
            'Humidity3pm': [humidity3pm],
            'Pressure9am': [pressure9am],
            'Pressure3pm': [pressure3pm],
            'Cloud9am': [cloud9am],
            'Cloud3pm': [cloud3pm],
            'Temp9am': [temp9am],
            'Temp3pm': [temp3pm],
            'RainToday': [raintoday],
            'RISK_MM': [risk_mm]}

    sample = pd.DataFrame(data, columns=['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall',
                                         'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed',
                                         'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm',
                                         'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                                         'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RISK_MM'])
    # sample pre-processing
    sample.drop(['RISK_MM'], axis=1, inplace=True)

    sample['Date'] = pd.to_datetime(sample['Date'])
    sample['Year'] = sample['Date'].dt.year
    sample['Month'] = sample['Date'].dt.month
    sample['Day'] = sample['Date'].dt.day
    sample.drop('Date', axis=1, inplace=True)

    nominal = []
    list_columns = sample.columns.tolist()
    for i in list_columns:
        if sample[i].dtypes == 'object':
            nominal.append(i)

    le = LabelEncoder()
    for i in nominal:
        sample[i] = le.fit_transform(sample[i])

    global temp_lg
    global temp_rd
    global temp_svm
    global temp_dt

    log_reg = load('LogisticRegression1.joblib')
    temp_lg=log_reg.predict(sample)[0]

    rf = load('RandomForest1.joblib')
    temp_rd=rf.predict(sample)[0]

    svm = load('SupportVectorMachine1.joblib')
    temp_svm=svm.predict(sample)[0]

    dt = load('DecisionTree1.joblib')
    temp_dt=dt.predict(sample)[0]

    print("tao da chay xong r")




class newwin:
    def __init__(self):
        self.top = Toplevel()
        self.top.title("halohalo")
        self.top.wm_geometry("1920x1080")
        imgpath = 'khoa1.png'
        img = Image.open(imgpath)
        photo1 = ImageTk.PhotoImage(img)
        self.frame = Canvas(self.top, width=1920, height=1080)
        self.frame.place(x=0, y=0)
        self.frame.create_image(0, 0, image=photo1, anchor=NW)
        #self.frame=Frame(self.top)
        #self.frame.pack(side=TOP)
        fontStyle = tkFont.Font(family="Helvetica", size=15, weight="bold")
        self.button2 = Button(self.frame, text="close windown",font = fontStyle, command=self.top.destroy)
        self.id=self.frame.create_window(800,780,width=200,height=50,window=self.button2)


# cai nay khoa pham code nha
class openwin():
    def __init__(self):
        def SVMFunc():
            if temp_svm==0.0:
                showinfo("NO")
            else :
                showinfo("Yes")

        def Logistic_Regression():
            if temp_lg==0.0:
                showinfo("NO")
            else :
                showinfo("YES")

        def RandomForest():
            if temp_rd==0.0:
                showinfo("NO")
            else :
                showinfo("YES")

        def DecisionTree():
            if temp_dt==0.0:
                showinfo("NO")
            else:
                showinfo("YES")


        def getValue():
            global counting
            a= str(self.combobox.get())
            list_test_1[counting]=a
            b=str(self.textbox.get("1.0", "end-1c"))
            list_test_2[counting]=b
            counting = counting + 1
            showinfo2(list_test_1,list_test_2,counting)

        def getValue1():
            global temp_location
            global temp_WindGustDir
            global temp_WindDir9am
            global temp_WindDir3pm
            global temp_RainToday

            temp_location= str(self.combobox_location.get())
            temp_WindGustDir=str(self.combobox_windgustdir.get())
            temp_WindDir9am=str(self.combobox_WindDir9am.get())
            temp_WindDir3pm=str(self.combobox_WindDir3pm.get())
            temp_RainToday=str(self.combobox_RainToday.get())

        def showinfo(temp_str):
            lable_3 = Label(self.canvas)
            lable_3['bg'] = "cyan"
            lable_3['text'] =temp_str
            lable_3['font']=fontStyle2
            self.canvas.create_window(1250,550,width=500,height=300,window=lable_3)


        def showinfo2(temp_str,temp_str_1,zz):
            s=""
            for i in range(0,zz):
                s=s+temp_str[i]+"          "+temp_str_1[i]+"\n"
            lable_3 = Label(self.canvas)
            lable_3['bg'] = "pink"
            lable_3['text'] = s
            lable_3['font'] = fontStyle1
            self.canvas.create_window(300, 550, width=400, height=400, window=lable_3)

        a=newwin()
        self.canvas=a.frame
        fontStyle = tkFont.Font(family="Helvetica", size=15, weight="bold")
        fontStyle1 = tkFont.Font(family="Helvetica", size=10, weight="bold")
        fontStyle2=tkFont.Font(family="Helvetica", size=13, weight="bold")
        #self.canvas.pack(side=TOP)
        self.combobox_WindDir9am=Combobox(self.canvas)
        self.combobox = Combobox(self.canvas)
        self.combobox_location=Combobox(self.canvas)
        self.combobox_windgustdir=Combobox(self.canvas)
        self.combobox_WindDir3pm=Combobox(self.canvas)
        self.combobox_RainToday=Combobox(self.canvas)


        items = ("Date","MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustSpeed","WindSpeed9am","WindSpeed9am","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm","RISK_MM")
        location=("Albury", "BadgerysCreek", "Cobar", "CoffsHarbour", "Moree",
"Newcastle", "NorahHead", "NorfolkIsland", "Penrith", "Richmond",
"Sydney", "SydneyAirport", "WaggaWagga", "Williamtown",
"Wollongong", "Canberra", "Tuggeranong", "MountGinini", "Ballarat",
"Bendigo", "Sale", "MelbourneAirport", "Melbourne", "Mildura",
"Nhil", "Portland", "Watsonia", "Dartmoor", "Brisbane", "Cairns",
"GoldCoast", "Townsville", "Adelaide", "MountGambier", "Nuriootpa",
"Woomera", "Albany", "Witchcliffe", "PearceRAAF", "PerthAirport",
"Perth", "SalmonGums", "Walpole", "Hobart", "Launceston",
"AliceSprings", "Darwin", "Katherine", "Uluru")
        windgustdir=("W", "WNW", "WSW", "NE", "NNW", "N", "NNE", "SW", "ENE", "SSE",
"S", "NW", "SE", "ESE", "E", "SSW")
        WindDir9am=("W", "WNW", "WSW", "NE", "NNW", "N", "NNE", "SW", "ENE", "SSE",
"S", "NW", "SE", "ESE", "E", "SSW")
        WindDir3pm=("W", "WNW", "WSW", "NE", "NNW", "N", "NNE", "SW", "ENE", "SSE",
"S", "NW", "SE", "ESE", "E", "SSW")
        RainToday=("Yes","No")

        self.combobox['values'] = items
        self.combobox.current(1)

        self.combobox_location['values']=location
        self.combobox_location.current(1)
        self.label_location=Label(self.canvas,text="location",font=fontStyle1)

        self.combobox_windgustdir['value']=windgustdir
        self.combobox_windgustdir.current(1)
        self.label_windgustdir=Label(self.canvas,text="windgustdir",font=fontStyle1)

        self.combobox_WindDir9am['value']=WindDir9am
        self.combobox_WindDir9am.current(1)
        self.label_WindDir9am=Label(self.canvas,text="WindDir9am",font=fontStyle1)

        self.combobox_WindDir3pm['value']=WindDir3pm
        self.combobox_WindDir3pm.current(1)
        self.label_WindDir3pm=Label(self.canvas,text="WindDir3pm",font=fontStyle1)

        self.combobox_RainToday['value']=RainToday
        self.combobox_RainToday.current(1)
        self.label_RainToday=Label(self.canvas,text="RainToday",font=fontStyle1)

        self.textbox=Text(self.canvas)
        self.buttonk=Button(self.canvas,text="khoa")
        self.button_get_value = Button(self.canvas, text="Get Value", command=getValue)
        self.button_show=Button(self.canvas,text="show info",command=showinfo)
        self.button_get_value_1=Button(self.canvas,text="Get Value",command=getValue1)
        self.button_run=Button(self.canvas,text="Run",command=sampletest)
        self.button_svm=Button(self.canvas,text="SVM",command=SVMFunc)
        self.button_logistic_regression=Button(self.canvas,text="Logistic Regression",command=Logistic_Regression)
        self.button_random_forest=Button(self.canvas,text="Random Forest",command=RandomForest)
        self.button_decision_tree=Button(self.canvas,text="Decision Tree",command=DecisionTree)
        self.button_show_1=Button(self.canvas,text="show",command=showketqua)


        self.label_1=Label(self.canvas)
        self.label_1['text']="các thuộc tính kiểu numeric"
        self.label_1['fg']="blue"
        self.label_1['font']=fontStyle

        self.label_2 = Label(self.canvas)
        self.label_2['text'] = "các thuộc tính kiểu nominal"
        self.label_2['fg'] = "blue"
        self.label_2['font'] = fontStyle



        #self.combobox
        #combobox.bind("<<ComboboxSelected>>", onChangeValue)
        #combobox.pack(side=LEFT)
        #self.canvas.id=canvas.create_window(500,500,width=100,height=100,window=combobox)
        self.id=self.canvas.create_window(80,170,width=130,height=30,window=self.combobox)
        self.id2 = self.canvas.create_window(300, 170, width=300, height=120,
                                             window=self.textbox)
        self.id3=self.canvas.create_window(300,300,width=100,height=70,window=self.button_get_value)
        self.id5=self.canvas.create_window(800,170,width=120,height=30,window=self.combobox_location)
        self.id6=self.canvas.create_window(950,170, width=120,height=30,window=self.combobox_windgustdir)
        self.id7=self.canvas.create_window(1100,170,width=120,height=30,window=self.combobox_WindDir9am)
        self.id8=self.canvas.create_window(1250,170,width=120,height=30,window=self.combobox_WindDir3pm)
        self.id9=self.canvas.create_window(300,50,width=300,height=50,window=self.label_1)
        self.id10=self.canvas.create_window(1000,50,width=300,height=50,window=self.label_2)
        self.id11=self.canvas.create_window(800,120,width=120,height=30,window=self.label_location)
        self.id12=self.canvas.create_window(950,120,width=120,height=30,window=self.label_windgustdir)
        self.id13=self.canvas.create_window(1100,120,width=120,height=30,window=self.label_WindDir9am)
        self.id14=self.canvas.create_window(1250,120,width=120,height=30,window=self.label_WindDir3pm)
        self.id15=self.canvas.create_window(1100,300,width=100,height=70,window=self.button_get_value_1)
        self.id16=self.canvas.create_window(800,480,width=150,height=30,window=self.button_svm)
        self.id17=self.canvas.create_window(800,530,width=150,height=30,window=self.button_logistic_regression)
        self.id18=self.canvas.create_window(800,580,width=150,height=30,window=self.button_random_forest)
        self.id19=self.canvas.create_window(800,630,width=150,height=30,window=self.button_decision_tree)
        self.id20=self.canvas.create_window(800,430,width=150,height=30,window=self.button_run)
        self.id21= self.canvas.create_window(1400,170,width=120,height=30,window=self.combobox_RainToday)
        self.id22=self.canvas.create_window(1400,120,width=120,height=30,window=self.label_RainToday)
        #self.id23=self.canvas.create_window(300,550,width=100,height=30,window=self.button_show_1)
        #self.id4=self.canvas.create_window(200,200,width=50,height=50,window=self.button_show)



# ai code test n thuoc tinh code o openwin2
class openwin2():
    def __init__(self):
        self.filePath = ""

        def c_open_file_old():
            rep = filedialog.askopenfilenames(
                filetypes=[
                    ("CSV", "*.csv"),
                    ("All files", "*")]
            )
            self.filePath = os.path.join(*rep)

        def decision_tree():
            modelPath = "DecisionTree1.joblib"
            dataFile = pd.read_csv(self.filePath)

            # Preprocessing:
            f = Preprocessing("output.csv")
            f.preprocessing("filetest.csv")
            file_data = pd.read_csv("output.csv")
            file_data_test = file_data.drop(['RainTomorrow'], axis=1)

            # ham chay dang bi loi

            dt = load(modelPath)
            result = dt.predict(file_data_test)
            df = pd.DataFrame(result)
            label = Label(self.canvas, text=df)
            label.config(font=("Helvetica", 17))
            label.place(x=850, y=150)

        def random_forest():
            modelPath = "RandomForest1.joblib"
            dataFile = pd.read_csv(self.filePath)

            # Preprocessing:
            f = Preprocessing("output.csv")
            f.preprocessing("filetest.csv")
            file_data = pd.read_csv("output.csv")
            file_data_test = file_data.drop(['RainTomorrow'], axis=1)

            rf = load(modelPath)
            result = rf.predict(file_data_test)
            df = pd.DataFrame(result)
            label = Label(self.canvas, text=df)
            label.config(font=("Helvetica", 17))
            label.place(x=850, y=150)

        def svm():
            modelPath = "SupportVectorMachine1.joblib"
            dataFile = pd.read_csv(self.filePath)

            # Preprocessing:
            f = Preprocessing("output.csv")
            f.preprocessing("filetest.csv")
            file_data = pd.read_csv("output.csv")
            file_data_test = file_data.drop(['RainTomorrow'], axis=1)

            svm = load(modelPath)
            result = svm.predict(file_data_test)

            df = pd.DataFrame(result)
            label = Label(self.canvas, text=df)
            label.config(font=("Helvetica", 17))
            label.place(x=850, y=150)

        def Logistic():
            modelPath = "LogisticRegression1.joblib"
            dataFile = pd.read_csv(self.filePath)

            # Preprocessing:
            f = Preprocessing("output.csv")
            f.preprocessing("filetest.csv")
            file_data = pd.read_csv("output.csv")
            file_data_test = file_data.drop(['RainTomorrow'], axis=1)

            logistic = load(modelPath)
            result = logistic.predict(file_data_test)

            df = pd.DataFrame(result)
            label = Label(self.canvas, text=df)
            label.config(font=("Helvetica", 17))
            label.place(x=850, y=150)

        b = newwin()
        self.canvas = b.frame

        self.label_Title = Label(self.canvas, text="Áp dụng mô hình với dữ liệu nhập từ File")
        self.label_Title.config(font=("Helvetica", 30))

        self.label_openFile = Label(self.canvas, text="Xin mời nhập File", font="Helvetica")
        self.label_openFile.config(font=("Helvetica", 17))
        self.label_feature = Label(self.canvas, text="Chọn mô hình", font="Helvetica")
        self.label_feature.config(font=("Helvetica", 17))
        self.label_result = Label(self.canvas, text="Kết quả dự đoán", font="Helvetica")
        self.label_result.config(font=("Helvetica", 17))
        self.openFileBtn = Button(self.canvas, text="open file", command=c_open_file_old)
        self.openFileBtn.config(font=("Helvetica", 17))
        self.DecisionTreeBtn = Button(self.canvas, text="Decision Tree", command=decision_tree)
        self.DecisionTreeBtn.config(font=("Helvetica", 17))
        self.RandomForestBtn = Button(self.canvas, text="Random Forest", command=random_forest)
        self.RandomForestBtn.config(font=("Helvetica", 17))
        self.SvmBtn = Button(self.canvas, text="Support Vector Machine", command=svm)
        self.SvmBtn.config(font=("Helvetica", 17))
        self.LogisticBtn = Button(self.canvas, text="Logistic Regression", command=Logistic)
        self.LogisticBtn.config(font=("Helvetica", 17))

        self.label_Title.place(x =350, y=25)
        self.label_openFile.place(x=100, y=150)
        self.openFileBtn.place(x=350, y=150)

        self.label_feature.place(x=100, y=250)
        self.DecisionTreeBtn.place(x=350, y=270)
        self.RandomForestBtn.place(x=350, y=320)
        self.SvmBtn.place(x=350, y=370)
        self.LogisticBtn.place(x=350, y=420)
        self.label_result.place(x=650, y=150)

# ai code huan luyen mo hinh code o cai openwin3
"""class openwin3():
    def __init__(self):
        newwin()
        print(list_test_1[0])"""



# ai code tien xu ly du lieu code o cai openwin4
class openwin4():
    def __init__(self):
        self.top = tk
        canvas1 = tk.Canvas(root, width=1920, height=1080, bg='white', relief='raised')
        canvas1.pack()

        def getCSV():
            global df

            import_file_path = filedialog.askopenfilename()
            df = pd.read_csv(import_file_path)

            dt = df

            # button1_show_features
            def myClick1():
                messagebox.showinfo("Result", df.columns.tolist())
                # myLabel1 = Label(canvas1, text=df.columns.tolist(), bg='white', fg='black', padx=80, pady=20)
                # myLabel1.pack()

            my_Button = Button(canvas1, text="SHOW FEATURES", command=myClick1, bg='cyan', fg='black', padx=55,
                               pady=20).place(x=70, y=150)
            canvas1.create_window(150, 150, window=my_Button)  # 2 cho

            # button2_drop_features

            def myClick2():

                def myClick3():
                    global df
                    str1 = E1.get()
                    df = df.drop(columns=[E1.get()])
                    messagebox.showinfo("Result", df.columns.tolist())

                    # .to_csv(self.output_file, index=False, header=True)

                top = Tk()
                top.geometry("400x200")
                top.configure(background='white')

                L1 = Label(top, text="Thuoc tinh", padx=40, pady=10)
                L1.pack(side=LEFT)
                E1 = Entry(top, bd=5)
                E1.pack(side=RIGHT)
                my_Button4 = Button(top, text="Confirm", command=myClick3, padx=40, pady=10)
                my_Button4.pack(side=BOTTOM)
                top.mainloop()

            my_Button1 = Button(canvas1, text="DROP FEATURES", command=myClick2, bg='cyan', fg='black', padx=55,
                                pady=20).place(x=280, y=150)
            canvas1.create_window(150, 150, window=my_Button1)  # 2 cho

            # Chuan hoa du lieu
            def myClick3():
                count_num = 0
                count_nom = 0
                numeric = []
                nominal = []
                list_columns = df.columns.tolist()
                for i in list_columns:
                    if df[i].dtypes == 'float64' or df[i].dtypes == 'int64':
                        numeric.append(i)
                    else:
                        nominal.append(i)
                # with null data in numeric columns, we can replace it by mean value of column
                for i in numeric:
                    mean = df[i].mean()
                    df[i].replace(to_replace=np.nan, value=mean, inplace=True)


                # with null data in nominal columns, we can replace it by popular string
                for i in nominal:
                    popular_str = df[i].mode()[0]
                    df[i].replace(to_replace=np.nan, value=popular_str, inplace=True)
                messagebox.showinfo("Result", "DONE")

            my_Button3 = Button(canvas1, text="STANDARDIZED", command=myClick3, bg='cyan', fg='black', padx=52,
                                pady=20).place(x=490, y=150)
            canvas1.create_window(150, 150, window=my_Button3)

            # FIX DAY
            def myClick4():
                df['Date'] = pd.to_datetime(df['Date'])
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                df.drop('Date', axis=1, inplace=True)
                messagebox.showinfo("Result", "Done")

            my_Button4 = Button(canvas1, text="FIX DAY", command=myClick4, bg='cyan', fg='black', padx=65,
                                pady=20).place(x=690, y=150)
            canvas1.create_window(150, 150, window=my_Button4)

            # IQR
            def myClick5():
                outliers = ['Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'Evaporation']
                text = " "
                for i in outliers:
                    old_min = df[i].min()
                    old_max = df[i].max()

                    IQR = df[i].quantile(0.75) - df[i].quantile(0.25)
                    lower_fence = df[i].quantile(0.25) - (IQR * 1.5)
                    upper_fence = df[i].quantile(0.75) + (IQR * 1.5)
                    df[i] = np.where(df[i] < lower_fence, lower_fence, df[i])
                    df[i] = np.where(df[i] > upper_fence, upper_fence, df[i])

                    new_min = df[i].min()
                    new_max = df[i].max()

                    text = text + "Thuoc tinh " + str(df[i].name) + "Min, max sau khi thay doi: min_old: " + str(
                        old_min) + ", max_old: " + str(
                        old_max) + " and new_min: " + str(new_min) + ", new_max: " + str(new_max) + "\n"

                messagebox.showinfo("Result", text)

            my_Button5 = Button(canvas1, text="IQR", command=myClick5, bg='cyan', fg='black', padx=68,
                                pady=20).place(x=865, y=150)
            canvas1.create_window(150, 150, window=my_Button5)

            # MIN_MAX
            def myClick6():
                global df
                scale = MinMaxScaler()
                df = pd.DataFrame(scale.fit_transform(df), columns=df.columns)
                messagebox.showinfo("Result", "Done")

            my_Button6 = Button(canvas1, text="MIN_MAX", command=myClick6, bg='cyan', fg='black', padx=60,
                                pady=20).place(x=1030, y=150)
            canvas1.create_window(150, 150, window=my_Button6)

            # ENCODER
            def myClick7():
                numeric = []
                nominal = []
                list_columns = df.columns.tolist()
                for i in list_columns:
                    if df[i].dtypes == 'float64' or df[i].dtypes == 'int64':
                        numeric.append(i)
                    else:
                        nominal.append(i)
                le = LabelEncoder()
                # data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
                for i in nominal:
                    print(str(i))
                    df[i] = le.fit_transform(df[i])

                messagebox.showinfo("Result", "DONE")

            my_Button7 = Button(canvas1, text="ENCODER", command=myClick7, bg='cyan', fg='black', padx=60,
                                pady=20).place(x=1215, y=150)
            canvas1.create_window(150, 150, window=my_Button7)

            # EXPORT
            def myClick8():
                df.to_csv('output.csv', index=False, header=True)
                messagebox.showinfo("Result", "Done")

            my_Button8 = Button(canvas1, text="EXPORT", command=myClick8, bg='cyan', fg='black', padx=70,
                                pady=20).place(x=1400, y=150)
            canvas1.create_window(150, 150, window=my_Button8)

            # Import du lieu

        browseButton_CSV = tk.Button(canvas1, text="      Import CSV File     ", command=getCSV, bg='green', fg='white',
                                     font=('helvetica', 12, 'bold')).place(x=250, y=80)
        canvas1.create_window(150, 150, window=browseButton_CSV)
        fontStyle = tkFont.Font(family="Helvetica", size=15, weight="bold")

        def Last_Click():
            canvas1.destroy()

        DT_Button = tk.Button(canvas1, text="      CLOSE     ", command=Last_Click, bg='green', fg='white',
                              font=('helvetica', 12, 'bold')).place(x=450, y=80)
        canvas1.create_window(150, 150, window=DT_Button)



#ai code phan danh gia code o day
class openwin5():
    def __init__(self):
        #a = newwin()
        #self.canvas = a.frame
        img = PhotoImage(file="D:\\giao dien mon hoc ham lone\modehard\khoa.png")

        def open_img():
            b = newwin()
            self.canvas1 = b.frame
            lable_3 = Label(self.canvas1)
            lable_3.config(image=img)
            # lable_3['bg'] = "cyan"
            # lable_3['text'] = "khoa"
            self.canvas.create_window(1250, 550, width=200, height=200, window=lable_3)

        a = newwin()
        self.canvas = a.frame
        self.button = Button(self.canvas, text="Accuracy train", command=open_img)
        self.id = self.canvas.create_window(300, 400, width=100, height=70, window=self.button)

        self.button = Button(self.canvas, text="Accuracy test", command=open_img)
        self.id = self.canvas.create_window(500, 400, width=100, height=70, window=self.button)

        self.button = Button(self.canvas, text="F1 score", command=open_img)
        self.id = self.canvas.create_window(700, 400, width=100, height=70, window=self.button)

        self.button = Button(self.canvas, text="Precision", command=open_img)
        self.id = self.canvas.create_window(900, 400, width=100, height=70, window=self.button)

        self.button = Button(self.canvas, text="Recall", command=open_img)
        self.id = self.canvas.create_window(1100, 400, width=100, height=70, window=self.button)

        self.button = Button(self.canvas, text="Time", command=open_img)
        self.id = self.canvas.create_window(1300, 400, width=100, height=70, window=self.button)

class CanvasButton:
    def __init__(self, canvas):
        self.canvas = canvas
        #self.number = IntVar()
        fontStyle  = tkFont.Font (family="Helvetica",size=36,weight="bold")
        self.lable = Label(canvas, text="appliction of team 17", font=fontStyle,bg='pink')

        self.button = Button(canvas, text="Test 1 thuoc tinh",command=openwin)
        self.button2= Button(canvas,text="Test n thuoc tinh",command=openwin2)
        #self.button3= Button(canvas,text='Huấn luyện mô hình phân ',command=openwin3)
        self.button4=Button(canvas,text=" Tiền xữ lý dữ liệu",command=openwin4)
        self.button5=Button(canvas,text="Đánh giá kết quả phân lớp theo độ đo",command=openwin5)



        self.id = canvas.create_window(750, 390, width=300, height=30,
                                       window=self.button)
        self.id2=canvas.create_window(750, 300 , width=500, height=50,
                                       window=self.lable)
        self.id3=canvas.create_window(750,440,width=300, height=30,window=self.button2)
        #self.id4=canvas.create_window(750,490,width=300, height=30,window=self.button3)
        self.id5=canvas.create_window(750,490,width=300,height=30,window=self.button4)
        self.id6=canvas.create_window(750,540,width=300,height=30,window=self.button5)



if __name__ == "__main__":
    root=Tk()
    root.wm_geometry("1920x1080")
    imgpath = 'khoa1.png'
    img = Image.open(imgpath)
    photo = ImageTk.PhotoImage(img)
    canvas = Canvas(root,width = 1920, height= 1080)
    canvas.place(x=0,y=0)
    canvas.create_image(0,0,image=photo,anchor=NW)
    CanvasButton(canvas)
    root.mainloop()