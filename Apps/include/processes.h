/*
 * processes.h
 *
 *  Created on: 25.1.2016
 *      Author: JV
 */

#ifndef PROCESSES_H_
#define PROCESSES_H_
#include <fstream>
#include "BioFW/Evaluation/Results.h"
#include "BioFW/Evaluation/Statistics/EERThreshFinder.h"


struct SortByResults {

		SortByResults(BioFW::Results & r) :
				results(r) {
			data = std::make_shared<SortMap>();
		}

		int eerErrors(const std::string & i) {
			if (data->find(i) == data->end()) {
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

		typedef std::map<std::string, int> SortMap;
		std::shared_ptr<SortMap> data;
};

BioFW::Results evaluateBlocks(ThermoCreator::Db & db, Blocks & blocks, int step = 1) {

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

	for (int i = 0; i < blocks.size(); i++) {

		Blocks tmp;
		for (int a = 0; a < step; a++) {
			if (blocks.empty()) {
				break;
			}
			IBlockPtr bl = blocks.front();
			blocks.pop_front();
			tmp.push_back(bl);
			//resultNames.push_back(bl->getName());
		}

		BioFW::BlockTraining<FaceRecord, FaceSample> trainer(std::make_shared<FaceSampleLoader>());
		for (auto b : tmp) {
			trainer.addBlock(b);
		}

		std::cout << "train start" << std::endl;
		trainer.ProgressChanged += Poco::delegate(&onProgress);

		trainer.train(trainDbSubset.getCollections().getRecords());

		trainer.ProgressChanged -= Poco::delegate(&onProgress);
		std::cout << "train ends" << std::endl;

		BioFW::BlockEvaluator<FaceRecord, FaceSample> evaluator(std::make_shared<FaceSampleLoader>());
		for (auto b : tmp) {
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

BestResult dumpResults(BioFW::Results & results, std::string filename) {

	std::vector<std::string> resultNames;
	BioFW::Results::Methods allmethods = results.getMethods();
	for (auto m : allmethods) {
		resultNames.push_back(m);
	}

	std::cout << "start sorting" << std::endl;

	SortByResults sr(results);
	std::sort(resultNames.begin(), resultNames.end(), sr);

	std::cout << "sorting done" << std::endl;
	std::cout << "start writing" << std::endl;

	std::ofstream os(filename);

	for (auto b : resultNames) {
		std::string name = b;
		os << "--------------------" << std::endl;
		os << name << ":" << std::endl;
		os << "--------------------" << std::endl;
		BioFW::Utils::dumpEEROverview(os, results.getMethodResult(name));
	}

	int cnt = 0;
	for (auto b : resultNames) {
		if (cnt++ < 2) {
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

	if (resultNames.empty()) {
		throw std::runtime_error("No results given");
	}

	BestResult br;
	br.name = resultNames.at(0);
	br.sum = sr.eerErrors(br.name);

	std::cout << "writing done" << std::endl;

	cv::waitKey(2000);

	return br;
}


#endif /* PROCESSES_H_ */
