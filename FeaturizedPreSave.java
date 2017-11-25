package org.hackathon.bottles;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;

/**
 * The TransferLearningHelper class allows users to "featurize" a dataset at specific intermediate vertices/layers of a model
 * This example demonstrates how to presave these
 * Refer to the "FitFromFeaturized" example for how to fit a model with these featurized datasets
 * @author susaneraly on 2/28/17.
 */
public class FeaturizedPreSave {
    private static final String featureFilenamePrefix = "bottles-";
	private static final Logger LOGGER = org.slf4j.LoggerFactory.getLogger(FeaturizedPreSave.class);
    private static final int trainPerc = 80;
    private static final int batchSize = 15;
    public static final String featurizeExtractionLayer = "fc1";

    public static final String featureFolder = Configuration.baseFolder() + "/features/bottles/";
    public static final String trainFolder = featureFolder + "trainFolder";
    public static final String testFolder = featureFolder + "testFolder";

    public static void main(String [] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        //import org.deeplearning4j.transferlearning.vgg16 and print summary
    	System.out.println("bla");
        LOGGER.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
        ZooModel zooModel = new VGG16();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        LOGGER.info(vgg16.summary());

        //use the TransferLearningHelper to freeze the specified vertices and below
        //NOTE: This is done in place! Pass in a cloned version of the model if you would prefer to not do this in place
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16, featurizeExtractionLayer);
        LOGGER.info(vgg16.summary());

        BottlesDataSetIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = BottlesDataSetIterator.trainIterator();
        DataSetIterator testIter = BottlesDataSetIterator.testIterator();

        int trainDataSaved = 0;
        while(trainIter.hasNext()) {
        	LOGGER.info("next trainIter");
            DataSet nextBlock = trainIter.next();
            LOGGER.info("featurize next block");
			DataSet currentFeaturized = transferLearningHelper.featurize(nextBlock);
			LOGGER.info("featurized next block");
            saveToDisk(currentFeaturized,trainDataSaved,true);
            trainDataSaved++;
        }

        int testDataSaved = 0;
        while(testIter.hasNext()) {
            DataSet currentFeaturized = transferLearningHelper.featurize(testIter.next());
            saveToDisk(currentFeaturized,testDataSaved,false);
            testDataSaved++;
        }

        LOGGER.info("Finished pre saving featurized test and train data");
    }

    private static void saveToDisk(DataSet currentFeaturized, int iterNum, boolean isTrain) {
        File fileFolder = isTrain ? new File(trainFolder): new File(testFolder);
        if (iterNum == 0) {
            fileFolder.mkdirs();
        }
        String fileName = featureFilenamePrefix + featurizeExtractionLayer + "-";
        fileName += isTrain ? "train-" : "test-";
        fileName += iterNum + ".bin";
        currentFeaturized.save(new File(fileFolder,fileName));
        LOGGER.info("Saved " + (isTrain?"train ":"test ") + "dataset #"+ iterNum + " to " + fileFolder + "/" + fileName) ;
    }
}
