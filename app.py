from flask import Flask,request,render_template,jsonify
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Cement=float(request.form.get('Cement')),
            Blast=float(request.form.get('Blast')),
            Fly_Ash = float(request.form.get('Fly_Ash')),
            Water = float(request.form.get('Water')),
            Superplasticizer = float(request.form.get('Superplasticizer')),
            Coarse_Aggregate = float(request.form.get('Coarse_Aggregate')),
            Fine_Aggregate = float(request.form.get('Fine_Aggregate')),
            Age = float(request.form.get('Age')),
           
           
            
           
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result=pred
        
        return render_template('result.html',final_result=result)





if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True,port=5000)