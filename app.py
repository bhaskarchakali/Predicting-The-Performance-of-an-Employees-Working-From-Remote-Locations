"""
app.py
"""
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index1.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Gender = (request.form['Gendar'])
        if(Gender == 'Female'):
            Gender = 0
        else:
            Gender = 1
        Marital_status = (request.form['Marital_status'])
        if Marital_status == 'single':
            Marital_status = 2
        elif(Marital_status == 'divorced'):
            Marital_status = 0
        else:
            Marital_Status = 1
        education = (request.form['Education'])
        if(education == 'Bachelor'):
            education = 0
        elif(education == 'Master'):
            education = 1
        elif(education == 'Phd'):
            education = 2
        else:
            education = 3
        Environmentsat = (request.form['Environmentsatisfaction'])
        if(Environmentsat == 'high'):
            Environmentsat = 0
        elif(Environmentsat == 'low'):
            Environmentsat = 1
        elif(Environmentsat == 'medium'):
            Environmentsat = 2
        else:
            Environmentsat = 3
        job_involvement = (request.form['JobInvolvement'])
        if(job_involvement == 'high'):
            job_involvement = 0
        elif(job_involvement == 'low'):
            job_involvement = 1
        elif(job_involvement == 'medium'):
            job_involvement = 2
        else:
            job_involvement = 3
        job_level = int(request.form['Job_Level'])
        Job_satisfaction = (request.form['Job_satisfaction'])
        if(Job_satisfaction == 'high'):
            Job_satisfaction= 0
        elif(Job_satisfaction == 'low'):
            Job_satisfaction = 1
        elif(Job_satisfaction == 'medium'):
            Job_satisfaction = 2
        else:
            Job_satisfaction = 3
        annual_income = int(request.form['Annual_income'])
        RelationshipSatisfaction = (request.form['RelationshipSatisfaction'])
        if(RelationshipSatisfaction == 'high'):
            RelationshipSatisfaction= 0
        elif(RelationshipSatisfaction == 'low'):
            RelationshipSatisfaction = 1
        elif(RelationshipSatisfaction == 'medium'):
            RelationshipSatisfaction = 2
        else:
            RelationshipSatisfaction = 3
        working_hrs = (request.form['Working_hrs_per_Day'])
        if (working_hrs == 'greaterthan9'):
            working_hrs = 1
        elif (working_hrs == 'equalto9'):
            working_hrs = 0
        else:
            working_hrs = 2
        experience = int(request.form['Experience'])
        training_time = int(request.form['Trainingtime'])

        WorkLifeBalance = (request.form['WorkLifeBalance'])
        if(WorkLifeBalance == 'bad'):
            WorkLifeBalance = 0
        elif(WorkLifeBalance == 'best'):
            WorkLifeBalance = 1
        elif(WorkLifeBalance == 'better'):
            WorkLifeBalance = 2
        else :
            WorkLifeBalance = 3
        BehaviouralCompetence = (request.form['BehaviouralCompetence'])
        if(BehaviouralCompetence == 'excellent'):
            BehaviouralCompetence = 0
        elif(BehaviouralCompetence == 'inadequate'):
            BehaviouralCompetence = 1
        elif(BehaviouralCompetence == 'poor'):
            BehaviouralCompetence = 2
        elif(BehaviouralCompetence =='satisfactory'):
            BehaviouralCompetence = 3
        elif(BehaviouralCompetence =='very_good'):
            BehaviouralCompetence = 4
        ontime_delivery = (request.form['On_timeDelivery'])
        if(ontime_delivery == 'excellent'):
            ontime_delivery = 0
        elif(ontime_delivery == 'good'):
            ontime_delivery = 1
        elif(ontime_delivery == 'poor'):
            ontime_delivery = 2
        else:
            ontime_delivery = 3
        TicketSolvingManagements = (request.form['TicketSolvingManagements'])
        if(TicketSolvingManagements == 'excellent'):
            TicketSolvingManagements = 0
        elif(TicketSolvingManagements == 'good'):
            TicketSolvingManagements = 1
        elif(TicketSolvingManagements == 'poor'):
            TicketSolvingManagements = 2
        else:
            TicketSolvingManagements = 3
        project_completion = int(request.form['ProjectCompletion'])
        WorkingFromHome = (request.form['WorkingFromHome'])
        if(WorkingFromHome == 'no'):
            WorkingFromHome = 0
        else:
            WorkingFromHome = 1
        Psycho_social_indicators = (request.form['Psycho_social_indicators'])
        if(Psycho_social_indicators == 'excellent'):
            Psycho_social_indicators = 0
        elif(Psycho_social_indicators == 'inadequate'):
            Psycho_social_indicators = 1
        elif(Psycho_social_indicators == 'poor'):
            Psycho_social_indicators = 2
        elif(Psycho_social_indicators =='satisfactory'):
            Psycho_social_indicators = 3
        elif(Psycho_social_indicators =='very_good'):
            Psycho_social_indicators = 4
        over_time = (request.form['Over_time'])
        if(over_time == 'no'):
            over_time = 0
        else:
            over_time = 1
        attendance = (request.form['Attendance'])
        if(attendance == 'good'):
            attendance = 0
        else:
            attendance = 1
        percent_salary_hike = int(request.form['PercentSalaryHike'])
        net_connection = (request.form['NetConnectivity'])
        if(net_connection == 'good'):
            net_connection = 0
        else:
            net_connection = 1
        department = (request.form['Department'])
        if(department == 'finance'):
            department = 0
        elif(department == 'HRM'):
            department = 1
        elif(department == 'IT'):
            department = 2
        elif(department =='RD'):
            department = 3
        elif(department =='sales'):
            department = 4
        position = (request.form['Position'])
        if(position == 'analyst'):
            position = 0
        elif(position =='Manager'):
            position = 1
        elif(position == 'Developer'):
            position = 2
        elif(position == 'executive'):
            position = 3
        elif(position =='HR'):
            position = 4
        elif position =='scientist':
            position = 5
        elif position == 'teamleader':
            position = 6

        prediction = model.predict([[Age, Gender, Marital_Status, education, Environmentsat, job_involvement, job_level, Job_satisfaction, annual_income, RelationshipSatisfaction, working_hrs, experience, training_time, WorkLifeBalance, BehaviouralCompetence, ontime_delivery, TicketSolvingManagements, project_completion, WorkingFromHome, Psycho_social_indicators, over_time, attendance, percent_salary_hike, net_connection, department, position]])
        #output = round(prediction[0], 2)
        if prediction < 0:
            return render_template('index1.html', prediction_texts="you did something wrong")
        elif prediction == 1:
            return render_template('index1.html', prediction_text="Performance Rating for Employee is {average}")
        elif prediction == 2:
            return render_template('index1.html', prediction_text="Performance Rating for Employee is {good}")
        elif prediction == 0:
            return render_template('index1.html', prediction_text="Performance Rating for Employee is {poor}")
    else:
        return render_template('index2.html')


if __name__ == "__main__":
    app.run(debug=True)