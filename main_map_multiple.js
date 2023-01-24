var data_parser = require('./data_parsing.js')
const math_extra = require("./math_extras")
const Regression = require('./regression_multiple.js')

let base_path = "./datasets"
let train_file_path = base_path+"/train_data.csv"
let val_file_path = base_path+"/val_data.csv"

// DATA STRUCTURE
// features
// fixed_acidity	volatile_acidity	citric_acid	residual_sugar	chlorides	free_sulfur_dioxide	total_sulfur_dioxide	density	pH	sulphates	alcohol

//target index 11
// quality

//feature a seleccionar 10 seria alcohol
const feature_index1 = 8
const feature_index2 = 10
//Nuestro target que es la calidad del vino
const target_index = 11

let model = new Regression(0.001, math_extra.mse)
let baseline_model = new Regression(-1, math_extra.mse)
baseline_model.predict = (x1, x2) => 5.64

//Read data
var train_data = data_parser.read_data(train_file_path)
//split feature and target
const train_x1 = train_data.map((x)=>x[feature_index1])
const train_x2 = train_data.map((x)=>x[feature_index2])
const train_y = train_data.map((x)=>x[target_index])

var val_data = data_parser.read_data(val_file_path)
//split feature and target
const val_x1 = val_data.map((x)=>x[feature_index1])
const val_x2 = val_data.map((x)=>x[feature_index2])
const val_y = val_data.map((x)=>x[target_index])

console.log("MAP")
console.time()
model_map = model.training_map(train_x1, train_x2, train_y, 5000)
console.timeEnd()

console.log("BASELINE EVALUACION MSE VALIDATION DATA: "+model.eval_map(val_x1, val_x2, val_y))
console.log("EVAL EVALUACION MSE VALIDATION DATA: "+baseline_model.eval_map(val_x1, val_x2, val_y))

for (let i=0; i<15; i++){
    console.log("Features: "+val_x1[i]+" "+val_x2[i])
    console.log("PREDICTION: "+model.predict(val_x1[i], val_x2[i]))
    console.log("REAL VALUE: "+val_y[i])
    console.log("---------------------------SAMPLE: ["+i+"]--------------------------------------")
}