import tkinter as tk
from PIL import ImageTk,Image
import Feature_Engineering as FE
import classification_model as CM
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fake Job Posting DataScience Project")
        self.root.geometry("1500x700+0+0")
        self.root.configure(bg="#2471A3")


        dataset_path="dataset\\fake_job_postings.csv"


        Data=FE.Feature_Engineering(dataset_path)




        x,y=CM.predict_data_and_target_class(Data)

        train_x,test_x,train_y,test_y=CM.split_x_y(x,y)



        # <------------------------------------------ Classification Model ---------------------------------------------------------->

        Logistics_Regression_y_prediction, Logistics_Regression_test_acc, Logistics_Regression_sensitivity, Logistics_Regression_specification, Logistics_Regression_precision, Logistics_Regression_F1_Score,Logistics_Regression_metrics = CM.Classification_model(
            LogisticRegression(), train_x, test_x, train_y, test_y)

        Decision_Tree_y_prediction, Decision_Tree_test_acc, Decision_Tree_sensitivity, Decision_Tree_specification, Decision_Tree_precision, Decision_Tree_F1_Score ,Decision_Tree_metrics= CM.Classification_model(
            DecisionTreeClassifier(), train_x, test_x, train_y, test_y)


        Forest_Regression_y_prediction, Forest_Regression_test_acc, Forest_Regression_sensitivity, Forest_Regression_specification, Forest_Regression_precision, Forest_Regression_F1_Score,Forest_Regression_metrics = CM.Classification_model(
            RandomForestClassifier(), train_x, test_x, train_y, test_y)

        SVM_y_prediction, SVM_test_acc, SVM_sensitivity, SVM_specification, SVM_precision, SVM_F1_Score ,SVM_metrics = CM.Classification_model(
            SVC(), train_x, test_x, train_y, test_y)

        #<---------------------------------------------------------------------------------------------------------------------------->

        # SVM support Vector Machine





        file_name = "images\\2.jpg"
        stdfilee = Image.open(file_name)
        stdshow = ImageTk.PhotoImage(stdfilee)
        imageshow = tk.Label(self.root, image=stdshow, border=0, bg="#01375d")
        imageshow.image = stdshow
        imageshow.place(relx=0.11, rely=0.0)

        self.heading = tk.Label(self.root,
                                text="Fake Job Postings DataScience Project",
                                fg="#fff",
                                bg="#2471A3", font=("Mongolian Baiti", "40"))
        self.heading.place(relx=0.2, rely=0.033)



        self.Task1 = tk.Frame(self.root, bg="#fff")
        self.Task1.place(rely=0.17, relx=0.03, relwidth=0.94, relheight=0.8)



        file_name = "images\\3.jpg"
        stdfilee = Image.open(file_name)
        stdshow = ImageTk.PhotoImage(stdfilee)
        imageshow = tk.Label(self.Task1, image=stdshow, border=0, bg="#01375d")
        imageshow.image = stdshow
        imageshow.place(relx=0.32, rely=0.15)

        self.heading = tk.Label(self.Task1, text="Classification Machine Learning Model", fg="#5F6062",
                                bg="#fff", font=("Mongolian Baiti", "22", "bold"))
        self.heading.place(relx=0.3, rely=0.02)

        self.tb_header_t2_7 = tk.Frame(self.Task1, bg="#2471A3")
        self.tb_header_t2_7.place(rely=0.17, relx=0.08, relwidth=0.85, relheight=0.07)

        self.l1 = tk.Label(self.tb_header_t2_7, text="Dataset", fg="#fff",
                           bg="#2471A3", font=("Mongolian Baiti", "15"))
        self.l1.place(relx=0.02, rely=0.1)

        self.l1 = tk.Label(self.tb_header_t2_7, text="Train Test Ratio", fg="#fff",
                           bg="#2471A3", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.13, rely=0.1)

        self.l1 = tk.Label(self.tb_header_t2_7, text="Accuracy Result", fg="#fff",
                           bg="#2471A3", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.28, rely=0.1)

        self.l1 = tk.Label(self.tb_header_t2_7, text="Specificity", fg="#fff",
                           bg="#2471A3", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.43, rely=0.1)

        self.l1 = tk.Label(self.tb_header_t2_7, text="Sensitivity", fg="#fff",
                           bg="#2471A3", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.58, rely=0.1)

        self.l1 = tk.Label(self.tb_header_t2_7, text="Precision", fg="#fff",
                           bg="#2471A3", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.7, rely=0.1)

        self.l1 = tk.Label(self.tb_header_t2_7, text="F1 score", fg="#fff",
                           bg="#2471A3", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.8, rely=0.1)

        self.l1 = tk.Label(self.tb_header_t2_7, text="Predicition error", fg="#fff",
                           bg="#2471A3", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.88, rely=0.1)






        self.tb_header1 = tk.Frame(self.Task1, bg="#ECF0F1", )
        self.tb_header1.place(rely=0.25, relx=0.08, relwidth=0.85, relheight=0.07)

        self.l1 = tk.Label(self.tb_header1, text="LR", fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.02, rely=0.1)

        self.l1 = tk.Label(self.tb_header1, text="0.3", fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.13, rely=0.1)

        self.l1 = tk.Label(self.tb_header1, text=str(round(Logistics_Regression_test_acc,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.28, rely=0.1)

        self.l1 = tk.Label(self.tb_header1, text=str(round(Logistics_Regression_sensitivity,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.43, rely=0.1)

        self.l1 = tk.Label(self.tb_header1, text=str(round(Logistics_Regression_specification,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.58, rely=0.1)

        self.l1 = tk.Label(self.tb_header1, text=str(round(Logistics_Regression_precision,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.7, rely=0.1)

        self.l1 = tk.Label(self.tb_header1, text=str(round(Logistics_Regression_F1_Score,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.8, rely=0.1)

        self.l1 = tk.Label(self.tb_header1, text=str(round(Logistics_Regression_metrics,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.88, rely=0.1)



        self.tb_header2 = tk.Frame(self.Task1, bg="#ECF0F1", )
        self.tb_header2.place(rely=0.33, relx=0.08, relwidth=0.85, relheight=0.07)

        self.l1 = tk.Label(self.tb_header2, text="DT", fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.02, rely=0.1)

        self.l1 = tk.Label(self.tb_header2, text="0.3", fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.13, rely=0.1)


        self.l1 = tk.Label(self.tb_header2, text=str(round(Decision_Tree_test_acc,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.28, rely=0.1)

        self.l1 = tk.Label(self.tb_header2, text=str(round(Decision_Tree_sensitivity,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.43, rely=0.1)

        self.l1 = tk.Label(self.tb_header2, text=str(round(Decision_Tree_specification,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.58, rely=0.1)

        self.l1 = tk.Label(self.tb_header2, text=str(round(Decision_Tree_precision,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.7, rely=0.1)

        self.l1 = tk.Label(self.tb_header2, text=str(round(Decision_Tree_F1_Score,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.8, rely=0.1)

        self.l1 = tk.Label(self.tb_header2, text=str(round(Decision_Tree_metrics,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.88, rely=0.1)


        self.tb_header3 = tk.Frame(self.Task1, bg="#ECF0F1", )
        self.tb_header3.place(rely=0.41, relx=0.08, relwidth=0.85, relheight=0.07)



        self.l1 = tk.Label(self.tb_header3, text="RF", fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.02, rely=0.1)

        self.l1 = tk.Label(self.tb_header3, text="0.3", fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.13, rely=0.1)


        self.l1 = tk.Label(self.tb_header3, text=str(round(Forest_Regression_test_acc,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.28, rely=0.1)

        self.l1 = tk.Label(self.tb_header3, text=str(round(Forest_Regression_sensitivity,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.43, rely=0.1)

        self.l1 = tk.Label(self.tb_header3, text=str(round(Forest_Regression_specification,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.58, rely=0.1)

        self.l1 = tk.Label(self.tb_header3, text=str(round(Forest_Regression_precision,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.7, rely=0.1)

        self.l1 = tk.Label(self.tb_header3, text=str(round(Forest_Regression_F1_Score,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.8, rely=0.1)

        self.l1 = tk.Label(self.tb_header3, text=str(round(Forest_Regression_metrics,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.88, rely=0.1)







        self.tb_header4 = tk.Frame(self.Task1, bg="#ECF0F1", )
        self.tb_header4.place(rely=0.49, relx=0.08, relwidth=0.85, relheight=0.07)



        self.l1 = tk.Label(self.tb_header4, text="SVM", fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.02, rely=0.1)

        self.l1 = tk.Label(self.tb_header4, text="0.3", fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.13, rely=0.1)


        self.l1 = tk.Label(self.tb_header4, text=str(round(SVM_test_acc,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.28, rely=0.1)

        self.l1 = tk.Label(self.tb_header4, text=str(round(SVM_sensitivity,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.43, rely=0.1)

        self.l1 = tk.Label(self.tb_header4, text=str(round(SVM_specification,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.58, rely=0.1)

        self.l1 = tk.Label(self.tb_header4, text=str(round(SVM_precision,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.7, rely=0.1)

        self.l1 = tk.Label(self.tb_header4, text=str(round(SVM_F1_Score,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.8, rely=0.1)

        self.l1 = tk.Label(self.tb_header4, text=str(round(SVM_metrics,2)), fg="#000",
                           bg="#ECF0F1", font=("Mongolian Baiti", "13"))
        self.l1.place(relx=0.88, rely=0.1)


        plt.figure(figsize=(3, 2))
        plt.plot(test_y, Logistics_Regression_y_prediction,color="blue",label="Logistics Regression")
        plt.plot(test_y, Decision_Tree_y_prediction, color="red", label="Decision Tree")
        plt.plot(test_y, Forest_Regression_y_prediction, color="green", label="Random Forest")
        plt.plot(test_y, SVM_y_prediction, color="black", label="SVM")
        plt.grid()
        plt.legend()
        plt.xlabel('Y Test')
        plt.ylabel('Predicted Y')
        plt.savefig("images//scatter.jpg")
        plt.close()

        plt.figure(figsize=(3, 2))
        plt.plot(x,y,color="blue")
        plt.grid()
        # plt.legend()
        plt.savefig("images//Visulization.jpg")
        plt.close()

        file_name = "images//scatter.jpg"
        stdfilee = Image.open(file_name)
        stdshow = ImageTk.PhotoImage(stdfilee)
        imageshow = tk.Label(self.Task1, image=stdshow, border=0, bg="#fff")
        imageshow.image = stdshow
        imageshow.place(relx=0.08, rely=0.58)

        self.heading = tk.Label(self.root,
                                text="Figure 1.1: Test y and prediction y Graph",
                                fg="#7A7A7A",
                                bg="#fff", font=("Mongolian Baiti", "11","bold"))
        self.heading.place(relx=0.1, rely=0.93)

        file_name = "images//Visulization.jpg"
        stdfilee = Image.open(file_name)
        stdshow = ImageTk.PhotoImage(stdfilee)
        imageshow = tk.Label(self.Task1, image=stdshow, border=0, bg="#fff")
        imageshow.image = stdshow
        imageshow.place(relx=0.7, rely=0.58)


        self.heading = tk.Label(self.root,
                                text="Figure 1.2: Dataset Graph",
                                fg="#7A7A7A",
                                bg="#fff", font=("Mongolian Baiti", "11", "bold"))

        self.heading.place(relx=0.7, rely=0.93)



















        self.root.mainloop()


if __name__ == '__main__':
    GUI()


