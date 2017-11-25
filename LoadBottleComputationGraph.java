package org.hackathon.bottles;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;

/**
 * A very simple example for saving and loading a ComputationGraph
 *
 * @author Alex Black
 */
public class LoadBottleComputationGraph {
	
	private static final Logger log = org.slf4j.LoggerFactory.getLogger(LoadBottleComputationGraph.class);

    public static void main(String[] args) throws Exception {
        //Load the model
    	File locationToSave = new File("/Users/kromes/Documents/cnn/trainedModels/bottlenet.zip");       //Where to save the network. Note: the file is in .zip format - can be opened externally
        ComputationGraph restored = ModelSerializer.restoreComputationGraph(locationToSave);
        
        BottlesDataSetIterator.setup(15,80);
        DataSetIterator trainIter = BottlesDataSetIterator.trainIterator();
        DataSetIterator testIter = BottlesDataSetIterator.testIterator();

        Evaluation eval;
        eval = restored.evaluate(testIter);
        log.info("Eval stats BEFORE fit.....");
        log.info(eval.stats() + "\n");


    }

}
