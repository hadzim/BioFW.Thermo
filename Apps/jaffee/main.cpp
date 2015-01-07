
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

template <
	class TInputRecord,
	class TBiometricSample
>
class ProjectTypes {
	public:
		typedef TInputRecord InputRecord;
		typedef TBiometricSample BiometricSample;
};

typedef BioFW::Record FaceRecord;



class JaffeCreator : public BioFW::IDatabaseCreator <FaceRecord> {
	public:
		virtual Db createDatabase(){
			std::string path = "c:/Data/JAFFE/";
			std::set<std::string> records;
			Poco::Glob::glob(path + "*.tiff", records);

			Db db;
			for (std::set<std::string>::iterator i = records.begin(); i != records.end(); i++){
				Poco::StringTokenizer st(*i, ".");

				BioFW::BiometricID bid(st[0], "Face");

				FaceRecord fr(*i, bid);
				db.addRecord(fr);
			}
			return db;
		}
};

typedef BioFW::MatSample FaceSample;


class FaceSampleLoader : public BioFW::IBiometricSampleLoader<FaceRecord, FaceSample> {
	public:
		virtual Sample createBiometricSample(const Record & record){
			FaceSample s;
			cv::Mat m = cv::imread(record.getSampleID(), cv::IMREAD_GRAYSCALE);
			cv::Mat dst;
			cv::equalizeHist(m, dst);
			s.mat() = dst;
			return s;
		}
};

typedef BioFW::MatFV FaceFeatureVector;


class FaceFeatureVectorExtractor : public BioFW::IFeatureVectorExtractor <FaceSample, FaceFeatureVector> {
	public:
		virtual FeatureVector extractFeatureVector(const Sample & input){
			FeatureVector fv;
			fv.mat() = input.mat();
			return fv;
		}
};

class FaceFeatureVectorComparator : public BioFW::IFeatureVectorComparator <FaceFeatureVector> {
	public:
		virtual BioFW::ComparisonScore computeComparisonScore(const ProbeFeatureVector & probe, const ReferenceFeatureVector & ref){
			cv::Mat diff;
			cv::absdiff(probe.mat(), ref.mat(), diff);
			return BioFW::ComparisonScore(cv::mean(diff).val[0]);
		}
};


typedef BioFW::Template<FaceFeatureVector> FaceTemplate;
typedef BioFW::TemplateCreator<FaceTemplate> FaceTemplateCreator;

typedef BioFW::Comparator<FaceFeatureVector, FaceTemplate> FaceComparator;

class FacePersTemplate : public BioFW::MemoryPersistentTemplate {
	public:
		FacePersTemplate(){

		}
		typedef std::shared_ptr <FacePersTemplate> Ptr;
		FaceTemplate faceTemplate;
};

class FaceTempalteSerializer : public BioFW::ITemplateSerializer <FaceTemplate, FacePersTemplate> {
	public:

		virtual PersistentTemplatePtr serialize(const Template & from){

			FacePersTemplate::Ptr fpt(new FacePersTemplate());
			fpt->faceTemplate = from;
			return fpt;
		}
		virtual Template deserialize(PersistentTemplatePtr fptr){
			return fptr->faceTemplate;
		}
};

void printDb(const JaffeCreator::Db & db, std::string name){
	std::cout << "-----------" << std::endl;
	std::cout << name << ":" << std::endl;
	BioFW::Utils::dumpDatabaseOverview(std::cout, db);
	std::cout << "-----------" << std::endl;
}

typedef BioFW::ProcessingBlock <FaceSample, FaceTemplate, FacePersTemplate> Block;
typedef BioFW::IProcessingBlock <FaceSample> IBlock;

void onProgress(BioFW::ProgressReport & report){
	std::cout << "\r";
	std::cout << "progress: " << report.getProgress() << "% " << report.getMessage() << "                ";
	std::cout << std::flush; //()<< std::endl;
}

int main(int argc, char **argv) {

	JaffeCreator creator;
	JaffeCreator::Db db = creator.createDatabase();

	printDb(db, "Full database");

	//create database subsets

	BioFW::TrainAndTestDatabaseSubsetCreator<FaceRecord> ttSubset(db);
	JaffeCreator::Db trainDbSubset = ttSubset.getTrainSubset();
	printDb(trainDbSubset, "Train database");


	JaffeCreator::Db testDbSubset = ttSubset.getTestSubset();


	BioFW::ReferenceAndProbeDatabaseSubsetCreator<FaceRecord> referenceAndProbeSubset(testDbSubset, 1);
	//template subset
	JaffeCreator::Db referenceDbSubset = referenceAndProbeSubset.getReferenceSubset();
	printDb(referenceDbSubset, "Reference database");
	//probe
	JaffeCreator::Db probeDbSubset = referenceAndProbeSubset.getProbeSubset();
	printDb(probeDbSubset, "Probe database");


	Block::Ptr simpleProcessingBlock(new Block(
			"simple",
			std::make_shared<FaceFeatureVectorExtractor>(),
			std::make_shared<FaceComparator>(std::make_shared<FaceFeatureVectorComparator>()),
			std::make_shared<FaceTemplateCreator>(),
			std::make_shared<FaceTempalteSerializer>()
	));

	typedef BioFW::WLDHistogramExtractor <FaceSample, FaceFeatureVector> WLDFE;

	Block::Ptr wldProcessingBlock(new Block(
			"wld",
			std::make_shared<WLDFE>(),
			std::make_shared<FaceComparator>(std::make_shared<WLDFE::DefaultComparator>()),
			std::make_shared<FaceTemplateCreator>(),
			std::make_shared<FaceTempalteSerializer>()
	));

	IBlock::Ptr nwldProcessingBlock(
			new BioFW::IScoreNormalizingProcessingBlock<FaceSample>(
					wldProcessingBlock,
					std::make_shared<BioFW::MinMaxScoreNormalizer>()
			)
	);

	IBlock::Ptr nzwldProcessingBlock(
				new BioFW::IScoreNormalizingProcessingBlock<FaceSample>(
						wldProcessingBlock,
						std::make_shared<BioFW::ZScoreNormalizer>()
				)
		);


	typedef BioFW::CombinedLBPHHistogramExtractor <FaceSample, FaceFeatureVector> CombLBPHFE;

		Block::Ptr clbpProcessingBlock(new Block(
				"CombinedLBPH",
				std::make_shared<CombLBPHFE>(),
				std::make_shared<FaceComparator>(std::make_shared<CombLBPHFE::DefaultComparator>()),
				std::make_shared<FaceTemplateCreator>(),
				std::make_shared<FaceTempalteSerializer>()
		));

		typedef BioFW::EigenfacesExtractor EigenFE;

		Block::Ptr eigenProcessingBlock(new Block(
			"Eigenfaces",
			std::make_shared<EigenFE>(),
			std::make_shared<FaceComparator>(std::make_shared<EigenFE::DefaultComparator>()),
			std::make_shared<FaceTemplateCreator>(),
			std::make_shared<FaceTempalteSerializer>()
		));

		typedef BioFW::FisherfacesExtractor FisherFE;

		Block::Ptr fisherProcessingBlock(new Block(
			"Fisherfaces",
			std::make_shared<FisherFE>(),
			std::make_shared<FaceComparator>(std::make_shared<FisherFE::DefaultComparator>()),
			std::make_shared<FaceTemplateCreator>(),
			std::make_shared<FaceTempalteSerializer>()
		));



		typedef BioFW::MultipleProcessingBlock<FaceSample> MultiBlock;

		MultiBlock::Ptr fusion(new BioFW::MultipleProcessingBlock<FaceSample>(
					"Fusion",
					std::make_shared<BioFW::SVMScoreFusion>()
				));

		MultiBlock::Ptr fusion2(new BioFW::MultipleProcessingBlock<FaceSample>(
			"Fusion2",
			std::make_shared<BioFW::SumScoreFusion>()
		));

		fusion->addBlock(simpleProcessingBlock);
		fusion->addBlock(eigenProcessingBlock);
		fusion->addBlock(wldProcessingBlock);
		fusion->addBlock(fisherProcessingBlock);

		fusion2->addBlock(simpleProcessingBlock);
		fusion2->addBlock(eigenProcessingBlock);
		fusion2->addBlock(wldProcessingBlock);
		fusion2->addBlock(fisherProcessingBlock);

	BioFW::BlockTraining<FaceRecord, FaceSample> trainer(std::make_shared<FaceSampleLoader>());
	trainer.addBlock(simpleProcessingBlock);
	//trainer.addBlock(eigenProcessingBlock);
	trainer.addBlock(nwldProcessingBlock);
	trainer.addBlock(nzwldProcessingBlock);
	//trainer.addBlock(fisherProcessingBlock);

	trainer.addBlock(fusion);
	trainer.addBlock(fusion2);

	std::cout << "train start" << std::endl;

	trainer.ProgressChanged += Poco::delegate(&onProgress);

	trainer.train(trainDbSubset.getCollections().getRecords());

	trainer.ProgressChanged -= Poco::delegate(&onProgress);

	std::cout << "train ends" << std::endl;

	BioFW::BlockEvaluator<FaceRecord, FaceSample> evaluator(std::make_shared<FaceSampleLoader>());
	evaluator.addBlock(simpleProcessingBlock);
	evaluator.addBlock(wldProcessingBlock);
	evaluator.addBlock(clbpProcessingBlock);
	//evaluator.addBlock(eigenProcessingBlock);
	evaluator.addBlock(nwldProcessingBlock);
	evaluator.addBlock(nzwldProcessingBlock);
	//evaluator.addBlock(fisherProcessingBlock);
	evaluator.addBlock(fusion);
	evaluator.addBlock(fusion2);

	 //database for templates
	BioFW::BiometricReferenceDatabase referenceDatabase;
	//progress report event
	evaluator.ProgressChanged += Poco::delegate(&onProgress);

	evaluator.extractTemplates(
		referenceDbSubset.getCollections().getRecords(),
		referenceDatabase
	);

	std::cout << std::endl;
	std::cout << "---------------------------------" << std::endl;
	std::cout << "All reference templates extracted" << std::endl;
	std::cout << "---------------------------------" << std::endl;
	std::cout << std::endl;

	BioFW::Results results;

	evaluator.evaluateRecords(
		probeDbSubset.getCollections().getRecords(),
		referenceDatabase,
		results
	);

	std::cout << std::endl;
	std::cout << "--------------------" << std::endl;
	std::cout << "All results computed" << std::endl;
	std::cout << "--------------------" << std::endl;
	std::cout << std::endl;

	evaluator.ProgressChanged -= Poco::delegate(&onProgress);

	std::set <std::string> names;
	names.insert(simpleProcessingBlock->getName());
	names.insert(wldProcessingBlock->getName());
	names.insert(clbpProcessingBlock->getName());
	//names.insert(eigenProcessingBlock->getName());
	names.insert(nwldProcessingBlock->getName());
	names.insert(nzwldProcessingBlock->getName());
	//names.insert(fisherProcessingBlock->getName());
	names.insert(fusion->getName());
	names.insert(fusion2->getName());


	for (auto name : names){
		std::cout << name << ":" << std::endl;
		BioFW::Utils::dumpEEROverview(std::cout, results.getMethodResult(name));

		cv::Mat image = BioFW::Utils::renderGenuineImpostorGraph(
				results.getMethodResult(name));
		cv::imshow(name, image);
	}

	cv::waitKey(0);






	return 0;
}
