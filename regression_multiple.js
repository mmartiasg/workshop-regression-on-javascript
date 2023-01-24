const math_extra = require("./math_extras")

class Regression{
    constructor(epsilon, metric){
        this.tita1 = (Math.random()*2-1)
        this.tita2 = (Math.random()*2-1)
        this.tita3 = (Math.random()*2-1)
        this.epsilon = epsilon
        this.metric = metric
    }

    reset_params(){
        this.tita1 = (Math.random()*2-1)
        this.tita2 = (Math.random()*2-1)
    }

    update_weights(y_pred, y, x1, x2){
        const N = x1.length
        
        // y = tita1 *x1 + tita2*x2 + tita3
        this.tita1 -= this.epsilon*math_extra.dotproduct(msubstract(y_pred, y), x1)/N
        this.tita2 -= this.epsilon*math_extra.dotproduct(msubstract(y_pred, y), x2)/N

        this.tita3 -= this.epsilon*math_extra.msubstract(y_pred, y).reduce((m, n) => m + n)/N
    }

    predict(x1, x2){
        return this.tita1 * x1 + this.tita2 * x2 + this.tita3
    }

    training_loop(train_x, train_y, MAX_EPOCHS){
        //Cantida de datos de entrenamiento
        const N = train_x.length

        //Estructuras auxiliares para actualizar los pesos en cada paso del entrenamiento
        let y_preds = new Array(N)
    
        for (let epoch=0; epoch<MAX_EPOCHS; epoch++){
            for (let i=0; i<N; i++){
                y_preds[i] = this.predict(train_x[i])
            }
    
            this.update_weights(y_preds, train_y, train_x)
    
            if (epoch%(MAX_EPOCHS/100)==0){
                console.log("----------------------"+"EPOCH: "+epoch+"--------------------------")
                console.log("TRAIN MSE: "+this.metric(y_preds, train_y))
            }
        }
    }

    training_map(train_x1, train_x2, train_y, MAX_EPOCHS){ 
        for (let epoch=0; epoch<MAX_EPOCHS; epoch++){
            let y_preds = (train_x1, train_x2).map((x, i) => this.predict(train_x1[i], train_x2[i]))    
            this.update_weights(y_preds, train_y, train_x1, train_x2)
    
            if (epoch%(MAX_EPOCHS/100)==0){
                console.log("----------------------"+"EPOCH: "+epoch+"--------------------------")
                console.log("TRAIN MSE: "+this.metric(y_preds, train_y))
            }
        }
    }

    eval_map(val_x1, val_x2, train_y){
        let y_preds = (val_x1, val_x2).map((x, i) => this.predict(val_x1[i], val_x2[i]))

        return this.metric(y_preds, train_y)
    }

}

module.exports = Regression
