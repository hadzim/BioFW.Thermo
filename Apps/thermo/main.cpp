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
#include "BioFW/Evaluation/Statistics/EERThreshFinder.h"

#include <fstream>

#include "BioFW/Exception.h"

void printDb(const ThermoCreator::Db & db, std::string name) {
	std::cout << "-----------" << std::endl;
	std::cout << name << ":" << std::endl;
	BioFW::Utils::dumpDatabaseOverview(std::cout, db);
	std::cout << "-----------" << std::endl;
}



void onProgress(BioFW::ProgressReport & report) {
	std::cout << "\r";
	std::cout << "progress: " << report.getProgress() << "% " << report.getMessage() << "                ";
	std::cout << std::flush; //()<< std::endl;
}


struct SortByResults {

		SortByResults(BioFW::Results & r) :
			results(r) {
			data = std::make_shared<SortMap>();
		}

		int eerErrors(const std::string & i){
			if (data->find(i) == data->end()){
				BioFW::Statistics::BasicStatistics bs(results.getMethodResult(i));
				BioFW::Statistics::ThreshedStatistics s(bs, std::make_shared<BioFW::Statistics::EERThreshFinder>());
				data->insert(std::make_pair(i, s.getFalseAcceptance() + s.getFalseRejection()));
			}
			return data->at(i);
		}

		bool operator()(const std::string & i, const std::string & j) {

			int vali = eerErrors(i);
			int valj = eerErrors(j);
			return vali < valj;
		}

		BioFW::Results & results;

		typedef std::map <std::string, int> SortMap;
		std::shared_ptr <SortMap> data;
};



BioFW::Results evaluateBlocks(ThermoCreator::Db & db, Blocks & blocks, int step = 1){

	BioFW::Results results;

	BioFW::TrainAndTestDatabaseSubsetCreator<FaceRecord> ttSubset(db);
	ThermoCreator::Db trainDbSubset = ttSubset.getTrainSubset();
	printDb(trainDbSubset, "Train database");

	ThermoCreator::Db testDbSubset = ttSubset.getTestSubset();

	BioFW::ReferenceAndProbeDatabaseSubsetCreator<FaceRecord> referenceAndProbeSubset(testDbSubset, 1);
	//template subset
	ThermoCreator::Db referenceDbSubset = referenceAndProbeSubset.getReferenceSubset();
	printDb(referenceDbSubset, "Reference database");
	//probe
	ThermoCreator::Db probeDbSubset = referenceAndProbeSubset.getProbeSubset();
	printDb(probeDbSubset, "Probe database");


	for (int i = 0; i < blocks.size(); i++){

		Blocks tmp;
		for (int a = 0; a < step; a++){
			if (blocks.empty()){
				break;
			}
			IBlockPtr bl = blocks.front();
			blocks.pop_front();
			tmp.push_back(bl);
			//resultNames.push_back(bl->getName());
		}



		BioFW::BlockTraining<FaceRecord, FaceSample> trainer(std::make_shared<FaceSampleLoader>());
		for (auto b : tmp){
			trainer.addBlock(b);
		}

		std::cout << "train start" << std::endl;
		trainer.ProgressChanged += Poco::delegate(&onProgress);

		trainer.train(trainDbSubset.getCollections().getRecords());

		trainer.ProgressChanged -= Poco::delegate(&onProgress);
		std::cout << "train ends" << std::endl;

		BioFW::BlockEvaluator<FaceRecord, FaceSample> evaluator(std::make_shared<FaceSampleLoader>());
		for (auto b : tmp){
			evaluator.addBlock(b);
		}

		//database for templates
		BioFW::BiometricReferenceDatabase referenceDatabase;
		//progress report event
		evaluator.ProgressChanged += Poco::delegate(&onProgress);

		evaluator.extractTemplates(referenceDbSubset.getCollections().getRecords(), referenceDatabase);
		std::cout << std::endl;
		std::cout << "---------------------------------" << std::endl;
		std::cout << "All reference templates extracted" << std::endl;
		std::cout << "---------------------------------" << std::endl;
		std::cout << std::endl;

		evaluator.evaluateRecords(probeDbSubset.getCollections().getRecords(), referenceDatabase, results);

		std::cout << std::endl;
		std::cout << "--------------------" << std::endl;
		std::cout << "All results computed" << std::endl;
		std::cout << "--------------------" << std::endl;
		std::cout << std::endl;

		evaluator.ProgressChanged -= Poco::delegate(&onProgress);

	}
	return results;
}

struct BestResult {
	int sum;
	std::string name;
};

BestResult dumpResults(BioFW::Results & results, std::string filename){

	std::vector <std::string> resultNames;
	BioFW::Results::Methods allmethods = results.getMethods();
	for (auto m : allmethods){
		resultNames.push_back(m);
	}

	std::cout << "start sorting" << std::endl;

	SortByResults sr(results);
	std::sort(resultNames.begin(), resultNames.end(), sr);

	std::cout << "sorting done" << std::endl;
	std::cout << "start writing" << std::endl;

	std::ofstream os(filename);


	for (auto b : resultNames){
		std::string name = b;
		os << "--------------------" << std::endl;
		os << name << ":" << std::endl;
		os << "--------------------" << std::endl;
		BioFW::Utils::dumpEEROverview(os, results.getMethodResult(name));
	}

	int cnt = 0;
	for (auto b : resultNames){
		if (cnt++ < 2){
			cv::Mat image = BioFW::Utils::renderGenuineImpostorGraph(results.getMethodResult(b));
			cv::imshow(b, image);
			std::stringstream bs;
			std::string a = b;
			a = Poco::replace(a, ".", "_");
			a = Poco::replace(a, " ", "_");
			a = Poco::replace(a, ":", "-");
			bs << a << ".png";
			std::cout << "fname: " << bs.str() << std::endl;
			cv::imwrite(bs.str(), image);
		}
	}

	if (resultNames.empty()){
		throw std::runtime_error("No results given");
	}

	BestResult br;
	br.name = resultNames.at(0);
	br.sum = sr.eerErrors(br.name);

	std::cout << "writing done" << std::endl;

	cv::waitKey(2000);

	return br;
}

typedef std::vector <std::string> DesiredNames;
typedef BioFW::MultipleProcessingBlock<FaceSample> MultiBlock;



void initLevel(DesiredNames & desiredNames, Blocks & desiredBlocks, int number){
	if (number <= 0) return;

	Blocks allblocks = createBlocks();

	for (auto b : allblocks){

		if (desiredNames[number - 1].find(b->getName()) != std::string::npos){
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

Blocks initCombinations(DesiredNames & desiredNames){
	std::cout << "INIT COMBINATIONS " << desiredNames.size() + 1 << std::endl;
	Blocks fusedBlocks;

	Blocks highestLevelBlocks = createBlocks();

	for (auto hl : highestLevelBlocks){
		MultiBlock::Ptr f(new BioFW::MultipleProcessingBlock<FaceSample>(std::make_shared<BioFW::SVMScoreFusion>()));

		Blocks lowerBlocks;
		initLevel(desiredNames, lowerBlocks, desiredNames.size());

		std::set <std::string> names;

		for (auto lb : lowerBlocks){
			f->addBlock(lb);
			names.insert(lb->getName());
		}

		//do not put same block again
		if (names.find(hl->getName()) != names.end()) continue;

		f->addBlock(hl);
		std::cout << "add: " << f->getName() << std::endl;
		fusedBlocks.push_back(f);
	}
	std::cout << "INIT COMBINATIONS done" << desiredNames.size() + 1 << std::endl;

	return fusedBlocks;
}

int main(int argc, char **argv) {
	try {
	ThermoCreator creator;
	ThermoCreator::Db db = creator.createDatabase();

	printDb(db, "Full database");

	//create database subsets

	DesiredNames names;
	for (int i = 1; i <= 7; i++){
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
	} catch (BioFW::BioFWException & e){
		std::cerr << "BioFW Exception: " << e.message() << std::endl;
	} catch (std::exception & e){
		std::cerr << "std::exception: " << e.what() << std::endl;
	}
	return 0;
}
