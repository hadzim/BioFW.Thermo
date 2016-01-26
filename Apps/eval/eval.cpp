/*
 * main.cpp
 *
 *  Created on: 25.1.2014
 *      Author: JV
 */

#include "BioFW/Database/Record.h"
#include "BioFW/Database/IDatabaseCreator.h"
#include "BioFW/Database/Database.h"

#include <iostream>
#include <memory>

#include "BioFW/Utils/Database.h"
#include <Poco/Glob.h>
#include "BioFW/Database/BiometricID.h"
#include <Poco/StringTokenizer.h>
#include <Poco/Delegate.h>
#include <Poco/String.h>
#include "BioFW/Database/Subset/ReferenceAndProbeDatabaseSubsetCreator.h"
#include "BioFW/Database/Subset/TrainAndTestDatabaseSubsetCreator.h"

#include <opencv2/opencv.hpp>
#include "BioFW/BiometricSample/IBiometricSampleLoader.h"
#include "BioFW/FeatureVector/IFeatureVectorExtractor.h"
#include "BioFW/Comparison/IFeatureVectorComparator.h"
#include "BioFW/Template/Template.h"
#include "BioFW/Template/TemplateCreator.h"
#include "BioFW/Template/MemoryPersistentTemplate.h"
#include "BioFW/Template/ITemplateSerializer.h"
#include "BioFW/Block/ProcessingBlock.h"
#include "BioFW/Block/MemoryProcessingBlock.h"
#include "BioFW/Comparsion/Comparator.h"
#include "BioFW/Evaluation/BlockEvaluator.h"
#include "BioFW/Evaluation/BlockTraining.h"
#include "BioFW/Database/Reference/BiometricReferenceDataRecord.h"
#include "BioFW/Utils/ResultsOverview.h"
#include "BioFW/Utils/Results.h"

#include "BioFW/FeatureVector/MatFV.h"
#include "BioFW/BiometricSample/MatSample.h"

#include "BioFW/FeatureVector/SpatialHistogram/WLDHistogramExtractor.h"
#include "BioFW/FeatureVector/SpatialHistogram/CombinedLBPHHistogramExtractor.h"
#include "BioFW/FeatureVector/Statistical/StatisticalExtractors.h"

#include "BioFW/Block/ScoreNormalization/ScoreNormalizingProcessingBlock.h"
#include "BioFW/Block/ScoreNormalization/MinMaxScoreNormalizer.h"
#include "BioFW/Block/ScoreNormalization/ZScoreNormalizer.h"

#include "BioFW/Block/ScoreFusion/MultipleProcessingBlock.h"
#include "BioFW/Block/ScoreFusion/SVMScoreFusion.h"
#include "BioFW/Block/ScoreFusion/SumScoreFusion.h"

#include "BioFW/FeatureVector/Statistical/ZScoreExtractor.h"
#include "BioFW/FeatureVector/ChainedFeatureExtractor.h"
#include "BioFW/FeatureVector/Filter/GaborFeatureExtractor.h"

#include "structures.h"
#include "primaryBanks.h"
#include "console.h"
#include "processes.h"

#include "BioFW/Evaluation/Statistics/EERThreshFinder.h"

#include <fstream>

#include "BioFW/Exception.h"


typedef std::vector<std::string> DesiredNames;
typedef BioFW::MultipleProcessingBlock<FaceSample> MultiBlock;

void initLevel(DesiredNames & desiredNames, Blocks & desiredBlocks, int number) {
	if (number <= 0)
		return;

	Blocks allblocks = createBlocks();

	for (auto b : allblocks) {

		if (desiredNames[number - 1].find(b->getName()) != std::string::npos) {
			desiredBlocks.push_back(b);
		}

		//desiredNamesif (b->getName())
	}

	/*
	 for (auto b : allblocks){

	 //desiredBlocks.clear();
	 //desiredBlocks.push_back(b);

	 MultiBlock::Ptr fusion(new BioFW::MultipleProcessingBlock<FaceSample>(std::make_shared<BioFW::SVMScoreFusion>()));

	 Blocks lowerLevelBlocks;
	 initLevel(desiredNames, lowerLevelBlocks, number - 1);

	 for (auto lowerLevelBlock : lowerLevelBlocks){
	 fusion->addBlock(lowerLevelBlock);
	 }

	 fusion->addBlock(b);


	 if (fusion->getName() == desiredNames[number - 1]){
	 desiredBlocks.push_back(b);
	 for (auto inblock : lowerLevelBlocks){
	 desiredBlocks.push_back(inblock);
	 }
	 return;
	 }

	 }
	 */
}

Blocks initCombinations(DesiredNames & desiredNames) {
	std::cout << "INIT COMBINATIONS " << desiredNames.size() + 1 << std::endl;
	Blocks fusedBlocks;

	Blocks highestLevelBlocks = createBlocks();

	for (auto hl : highestLevelBlocks) {
		MultiBlock::Ptr f(new BioFW::MultipleProcessingBlock<FaceSample>(std::make_shared<BioFW::SVMScoreFusion>()));

		Blocks lowerBlocks;
		initLevel(desiredNames, lowerBlocks, desiredNames.size());

		std::set<std::string> names;

		for (auto lb : lowerBlocks) {
			f->addBlock(lb);
			names.insert(lb->getName());
		}

		//do not put same block again
		if (names.find(hl->getName()) != names.end())
			continue;

		f->addBlock(hl);
		std::cout << "add: " << f->getName() << std::endl;
		fusedBlocks.push_back(f);
	}
	std::cout << "INIT COMBINATIONS done" << desiredNames.size() + 1 << std::endl;

	return fusedBlocks;
}

void evaluate(ThermoCreator::Db & db) {
	//create database subsets

	DesiredNames names;
	for (int i = 1; i <= 3; i++) {
		std::stringstream s;
		s << "results" << i << ".txt";
		Blocks blocks = initCombinations(names);
		BioFW::Results res = evaluateBlocks(db, blocks, 10);
		BestResult best = dumpResults(res, s.str());
		names.push_back(best.name);
		std::cout << "Best in iteration " << i << " is " << best.name << " Errors: " << best.sum << std::endl;
	}

	/*

	 typedef BioFW::ChainedFeatureExtractor<EigenFE, BioFW::ZScoreExtractor> ZPCA;
	 ZPCA::Ptr zpca(
	 new ZPCA(
	 EigenFE::Ptr(new EigenFE(0.95)),
	 std::make_shared<BioFW::ZScoreExtractor>()
	 )
	 );

	 Block::Ptr zpcaBlock(
	 new Block("ZPCA", wrap<ZPCA>(FaceSample::Warp2D_Global, zpca), std::make_shared < FaceComparator > (std::make_shared<EigenFE::DefaultComparator>()), std::make_shared<FaceTemplateCreator>(),
	 std::make_shared<FaceTempalteSerializer>()));


	 Block::Ptr gaborBlock(
	 new Block("Gabor", wrap<BioFW::GaborFeatureExtractor>(FaceSample::Warp2D_Global, BioFW::GaborFeatureExtractor::Ptr(new BioFW::GaborFeatureExtractor(4, 1, 1))), std::make_shared < FaceComparator > (std::make_shared<FaceFeatureVectorComparator>()), std::make_shared<FaceTemplateCreator>(),
	 std::make_shared<FaceTempalteSerializer>()));


	 typedef BioFW::FisherfacesExtractor FisherFE;

	 Block::Ptr fisherProcessingBlock(
	 new Block("LDA", wrap<FisherFE>(FaceSample::Warp2D_Global), std::make_shared < FaceComparator > (std::make_shared<FisherFE::DefaultComparator>()), std::make_shared<FaceTemplateCreator>(),
	 std::make_shared<FaceTempalteSerializer>()));







	 blocks.push_back(clbpProcessingBlock);

	 blocks.push_back(fisherProcessingBlock);
	 blocks.push_back(zpcaBlock);
	 blocks.push_back(gaborBlock);
	 */

	/*
	 typedef BioFW::MultipleProcessingBlock<FaceSample> MultiBlock;

	 MultiBlock::Ptr fusion(new BioFW::MultipleProcessingBlock<FaceSample>("Fusion SVM", std::make_shared<BioFW::SVMScoreFusion>()));

	 MultiBlock::Ptr fusion2(new BioFW::MultipleProcessingBlock<FaceSample>("Fusion Simple", std::make_shared<BioFW::SumScoreFusion>()));

	 fusion->addBlock(simpleProcessingBlock);
	 //fusion->addBlock(clbpProcessingBlock);
	 //fusion->addBlock(eigenProcessingBlock);
	 fusion->addBlock(wldProcessingBlock);
	 //fusion->addBlock(fisherProcessingBlock);

	 fusion2->addBlock(simpleProcessingBlock);
	 fusion2->addBlock(clbpProcessingBlock);
	 //fusion2->addBlock(eigenProcessingBlock);
	 fusion2->addBlock(wldProcessingBlock);
	 //fusion2->addBlock(fisherProcessingBlock);
	 */

	cv::waitKey(0);

}

int main(int argc, char **argv) {
	try {
		ThermoCreator creator;
		ThermoCreator::Db db = creator.createDatabase();

		printDb(db, "Full database");


		evaluate(db);

	} catch (BioFW::BioFWException & e) {
		std::cerr << "BioFW Exception: " << e.message() << std::endl;
	} catch (std::exception & e) {
		std::cerr << "std::exception: " << e.what() << std::endl;
	}
	return 0;
}
