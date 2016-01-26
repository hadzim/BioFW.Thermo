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

#include "BioFW/Evaluation/Statistics/EERThreshFinder.h"

#include <fstream>

#include "BioFW/Exception.h"
#include "processes.h"

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

IBlockPtr createBestBlock() {

	// {{PG+wld + Avg} {PG+CombinedLBPH + Avg} fus:SVM}

	std::cout << "best block" << std::endl;
	MultiBlock::Ptr f(new BioFW::MultipleProcessingBlock<FaceSample>(std::make_shared<BioFW::SVMScoreFusion>()));

	Blocks basicBlocks = createBlocks();

	for (auto b : basicBlocks) {
		std::string name = b->getName();
		if (name == "PG+wld") {
			f->addBlock(b);
		}
		if (name == "PG+CombinedLBPH") {
			f->addBlock(b);
		}
		std::cout << name << std::endl;
	}

	return f;
}

Blocks allBlocks() {
	Blocks basicBlocks = createBlocks();
	basicBlocks.push_back(createBestBlock());
	return basicBlocks;
}

cv::Mat loadImage(std::string file){
	cv::Mat firstImage = cv::imread(file, CV_LOAD_IMAGE_GRAYSCALE);

	int w = firstImage.cols;
	int h=  firstImage.rows;

	cv::Mat oo(cv::Size(w, h), CV_8UC3);

	for (int x = 0; x < w; x++){
		for (int y = 0; y < h; y++){
			unsigned char color =firstImage.at<unsigned char>(cv::Point(x,y));
			double c = color * 180.0 / 190.0;
			oo.at<cv::Vec3b>(cv::Point(x,y)) = cv::Vec3b(c, 255, color);
		}
	}

	cv::cvtColor(oo, oo, CV_HSV2BGR);
	return oo;
}

static void textxy(cv::Mat & m, int xpos, int ypos, std::string text, cv::Scalar color = cv::Scalar(255, 255, 255), double scale = 1) {

	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = scale;
	int thickness = 1;

	cv::putText(m, text, cv::Point(xpos, ypos), fontFace, fontScale, color, thickness, 8);
}

void verify(ThermoCreator & creator, ThermoCreator::Db & db) {
	IBlockPtr block = createBestBlock();
	Blocks blocks;
	blocks.push_back(block);
	evaluateBlocks(db, blocks, 10);

	ThermoCreator::Db pureDb = creator.createPureDatabase();

	auto allRecords = pureDb.getCollections().getRecords();

	auto reference = allRecords.begin();
	auto probe = allRecords.begin();
	probe++;

	while (true) {
		cv::Mat final(cv::Size(1100, 350), CV_8UC3);

		cv::Mat firstImage = loadImage(reference->getSampleID()); // cv::imread(reference->getSampleID(), CV_LOAD_IMAGE_COLOR);
		cv::Mat secondImage = loadImage(probe->getSampleID()); //cv::imread(probe->getSampleID(), CV_LOAD_IMAGE_COLOR);

		cv::Mat cpyFirst = final(cv::Rect(0, 0, firstImage.cols, firstImage.rows));
		cv::Mat cpySecond = final(cv::Rect(400, 0, secondImage.cols, secondImage.rows));

		firstImage.copyTo(cpyFirst);
		secondImage.copyTo(cpySecond);

		textxy(final, 80, 320, "q/w - prev/next");
		textxy(final, 480, 320, "a/s - prev/nex");

		//perform matching
		FaceSampleLoader ll;
		auto refSample = ll.createBiometricSample(*reference);
		auto probeSample = ll.createBiometricSample(*probe);
		auto refTemplate = block->extract(refSample);
		//auto probeTemplate = block->extract(probeSample);

		block->resetTemplates();
		auto ti = block->pushTemplate(refTemplate);
		block->setInputData(probeSample);
		BioFW::ComparisonScore score = block->computeComparisonScore(ti);
		double dscore = score.getScore();
		std::cout << "Score: " << score.getScore() << std::endl;

		if (dscore > 0) {
			textxy(final, 810, 100, "Accepted", cv::Scalar(0, 255, 0), 3);
		} else {
			textxy(final, 810, 100, "Rejected", cv::Scalar(0, 0, 255), 3);
		}

		textxy(final, 840, 150, "Score: " + Poco::NumberFormatter::format(dscore, 2));

		//std::cout << reference->getSampleID() << std::endl;
		cv::imshow("Verification Mode", final);
		int key = cv::waitKey(0);
		if (key == 'w') {
			if (reference + 1 != allRecords.end()) {
				reference++;
			}
		} else if (key == 'q') {
			if (reference != allRecords.begin()) {
				reference--;
			}
		} else if (key == 's') {
			if (probe + 1 != allRecords.end()) {
				probe++;
			}
		} else if (key == 'a') {
			if (probe != allRecords.begin()) {
				probe--;
			}
		}
	}

}


void identify(ThermoCreator & creator, ThermoCreator::Db & db) {
	IBlockPtr block = createBestBlock();
	Blocks blocks;
	blocks.push_back(block);
	evaluateBlocks(db, blocks, 10);


	ThermoCreator::Db pureDb = creator.createPureDatabase();

	auto allRecords = pureDb.getCollections().getRecords();

	//template subset
	BioFW::ReferenceAndProbeDatabaseSubsetCreator<FaceRecord> referenceAndProbeSubset(db, 1);
	ThermoCreator::Db referenceDbSubset = referenceAndProbeSubset.getReferenceSubset();
	printDb(referenceDbSubset, "Reference database");
	//probe
	ThermoCreator::Db probeDbSubset = referenceAndProbeSubset.getProbeSubset();
	printDb(probeDbSubset, "Probe database");


	block->resetTemplates();
	std::map <std::string, int> memDb;

	FaceSampleLoader ll;
	for (auto r : referenceDbSubset.getCollections().getRecords()){

		auto refSample = ll.createBiometricSample(r);
		auto refTemplate = block->extract(refSample);
		auto ti = block->pushTemplate(refTemplate);

		memDb[r.getSampleID()] = ti;
	}

	auto probe = allRecords.begin();
	probe++;

	while (true) {
		cv::Mat final(cv::Size(880, 530), CV_8UC3);

		cv::Mat secondImage = loadImage(probe->getSampleID());

		cv::Mat cpySecond = final(cv::Rect(0, 0, secondImage.cols, secondImage.rows));

		secondImage.copyTo(cpySecond);

		textxy(final, 400, 100, "q/w - prev/next");


		auto probeSample = ll.createBiometricSample(*probe);
		block->setInputData(probeSample);

		std::map <double, std::string> scores;

		for (auto mem : memDb){
			BioFW::ComparisonScore score = block->computeComparisonScore(mem.second);
			double dscore = score.getScore();
			scores[dscore] = mem.first;
		}


		int cnt = 0;
		for (auto it = scores.rbegin(); it != scores.rend(); it++){
			cv::Mat aimage = loadImage(it->second);
			cv::resize(aimage, aimage, cv::Size(aimage.cols/2, aimage.rows/2));
			cv::Mat cpySecond = final(cv::Rect(10+cnt*165, 350, aimage.cols, aimage.rows));

			aimage.copyTo(cpySecond);

			std::string sscore = Poco::NumberFormatter::format(it->first, 2);
			if (it->first > 0) {
				textxy(final, 20 + cnt*165, 300, "Accepted", cv::Scalar(0, 255, 0));
			} else {
				textxy(final, 20 + cnt*165, 300, "Rejected", cv::Scalar(0, 0, 255));
			}
			textxy(final, 20 + cnt*165, 325, sscore);


			cnt++;
			if (cnt > 4) break;
		}


		//std::cout << reference->getSampleID() << std::endl;
		cv::imshow("Identification Mode", final);
		int key = cv::waitKey(0);
		if (key == 'w') {
			if (probe + 1 != allRecords.end()) {
				probe++;
			}
		} else if (key == 'q') {
			if (probe != allRecords.begin()) {
				probe--;
			}
		}
	}

}


int main(int argc, char **argv) {
	try {
		ThermoCreator creator;
		ThermoCreator::Db db = creator.createDatabase();

		printDb(db, "Full database");

		std::vector<std::string> args;
		for (int i = 0; i < argc; i++) {
			args.push_back(argv[i]);
		}

		if (args.size() == 1) {
			std::cout << "help:" << std::endl;
			std::cout << "'[exe] -evaluate' - run algorithmic evaluation" << std::endl;
			std::cout << "'[exe] -verify' - compare two samples and return verification score" << std::endl;
			std::cout << "'[exe] -identify' - find best candidates in given database" << std::endl;
		}

		if (args.size() > 1) {
			if (args.at(1) == "-evaluate") {
				//evaluate(db);

				DesiredNames names;
				std::stringstream s;
				s << "resultsBest.txt";
				Blocks blocks = allBlocks();
				BioFW::Results res = evaluateBlocks(db, blocks, 10);
				BestResult best = dumpResults(res, s.str());
				names.push_back(best.name);
				std::cout << "Best in iteration " << " is " << best.name << " Errors: " << best.sum << std::endl;

				cv::waitKey(0);
			}

			if (args.at(1) == "-verify") {
				verify(creator, db);
			}

			if (args.at(1) == "-identify") {
				identify(creator, db);
			}
		}



	} catch (BioFW::BioFWException & e) {
		std::cerr << "BioFW Exception: " << e.message() << std::endl;
	} catch (std::exception & e) {
		std::cerr << "std::exception: " << e.what() << std::endl;
	}
	return 0;
}
