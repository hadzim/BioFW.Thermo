/*
 * structures.h
 *
 *  Created on: 12.3.2014
 *      Author: JV
 */

#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include <vector>
#include "BioFW/Database/Record.h"
#include "BioFW/Database/IDatabaseCreator.h"
#include "BioFW/FeatureVector/IFeatureVectorExtractor.h"
#include "structures.h"
#include "BioFW/Template/MemoryPersistentTemplate.h"
#include "BioFW/Template/Template.h"
#include "BioFW/Template/TemplateCreator.h"
#include "BioFW/Comparsion/Comparator.h"
#include <deque>


typedef BioFW::Record FaceRecord;

class ThermoCreator: public BioFW::IDatabaseCreator<FaceRecord> {
	public:
		virtual Db createDatabase() {
			Db db;

			addFolder(db, "c:/Data/DB/original/fitClean/");
			addFolder(db, "c:/Data/DB/original/notredame/");
			addFolder(db, "c:/Data/DB/original/germany/");
			addFolder(db, "c:/Data/DB/original/equinox/");



			return db;
		}

		void addFolder(Db & db, std::string path){
			std::set<std::string> records;
			Poco::Glob::glob(path + "*.bmp", records);

			BioFW::BiometricID::Set ids;

			for (std::set<std::string>::iterator i = records.begin(); i != records.end(); i++) {



				Poco::StringTokenizer st(*i, ".-");

				BioFW::BiometricID bid(st[0], "Face");


				ids.insert(bid);
				if (path.find("notredame") != std::string::npos){
					if (ids.size() < 78){
						continue;
					}
				} else if (path.find("fit") != std::string::npos){
					if (ids.size() > 12){
						std::cout << "break fit" << std::endl;
						break;
					}
				} else {
					if (ids.size() > 27){
						std::cout << "break other" << std::endl;
						break;
					}
				}


				FaceRecord fr(*i, bid);
				db.addRecord(fr);
			}

			std::cout << "db " << path << " size: " << ids.size() << std::endl;
		}
};

class FaceSample {
	public:
		enum ImageType {
			Project3D, Project3D_Global, Project3D_Local, Warp2D, Warp2D_Global, Warp2D_Local,
		};

		typedef std::map<ImageType, BioFW::MatSample> Images;
		Images images;

		BioFW::MatSample getSample(ImageType t) const {
			if (images.find(t) == images.end()) {
				throw std::runtime_error("non existing image");
			}
			return images.at(t);
		}
		void addSample(ImageType t, BioFW::MatSample s) {
			if (s.mat().empty()) {
				throw std::runtime_error("input sample is empty");
			}
			images[t] = s;
		}
};

template<class FeatureVectorExtractor>
class FaceFeatureExtractorWrapper: public BioFW::IFeatureVectorExtractor<FaceSample, typename FeatureVectorExtractor::FeatureVector> {
	public:
		typedef typename FeatureVectorExtractor::FeatureVector FeatureVector;
		typedef FaceSample Sample;
		typedef std::vector<Sample> Samples;

		FaceFeatureExtractorWrapper(typename FeatureVectorExtractor::Ptr fextractor, FaceSample::ImageType t) :
				fextractor(fextractor), t(t) {
		}
		virtual ~FaceFeatureExtractorWrapper() {

		}

		FeatureVector extractFeatureVector(const Sample & input) {
			BioFW::MatSample ms = input.getSample(t);
			return fextractor->extractFeatureVector(ms);
		}

		virtual void train(const Samples & inputs, const BioFW::Labels & labels) {
			std::vector<BioFW::MatSample> samples;
			for (auto s : inputs) {
				samples.push_back(s.getSample(t));
			}
			fextractor->train(samples, labels);
		}

	private:

		typename FeatureVectorExtractor::Ptr fextractor;
		FaceSample::ImageType t;
};

class FaceSampleLoader: public BioFW::IBiometricSampleLoader<FaceRecord, FaceSample> {
	public:
		virtual Sample createBiometricSample(const Record & record) {

			FaceSample s;

			std::map<FaceSample::ImageType, std::string> pathReplacements;

			pathReplacements[FaceSample::Project3D] = "\\Project3D\\_images\\";
			pathReplacements[FaceSample::Project3D_Global] = "\\Project3D\\Global\\_images\\";
			pathReplacements[FaceSample::Project3D_Local] = "\\Project3D\\Local\\_images\\";

			pathReplacements[FaceSample::Warp2D] = "\\Warp2D\\_images\\";
			pathReplacements[FaceSample::Warp2D_Global] = "\\Warp2D\\Global\\_images\\";
			pathReplacements[FaceSample::Warp2D_Local] = "\\Warp2D\\Local\\_images\\";

			std::string rpl = "\\";

			for (auto p : pathReplacements) {
				BioFW::MatSample matsample;

				std::string path = record.getSampleID();
				path = Poco::replace(path, "bmp", "png");
				path = Poco::replace(path, "original", "processed");

				path = Poco::replace(path, rpl, p.second, path.rfind('\\'));

				//std::cout << "create sample: " << path << std::endl;

				matsample.mat() = cv::imread(path, cv::IMREAD_GRAYSCALE);

				if (matsample.mat().empty()) {
					throw std::runtime_error(path);
				}

				s.addSample(p.first, matsample);
			}
			return s;
		}
};

typedef BioFW::MatFV FaceFeatureVector;

class FaceFeatureVectorExtractor: public BioFW::IFeatureVectorExtractor<BioFW::MatSample, FaceFeatureVector> {
	public:
		virtual FeatureVector extractFeatureVector(const Sample & input) {
			FeatureVector fv;
			fv.mat() = input.mat();
			return fv;
		}
};

class FaceFeatureVectorComparator: public BioFW::IFeatureVectorComparator<FaceFeatureVector> {
	public:
		virtual BioFW::ComparisonScore computeComparisonScore(const ProbeFeatureVector & probe, const ReferenceFeatureVector & ref) {
			cv::Mat diff;
			cv::absdiff(probe.mat(), ref.mat(), diff);
			return BioFW::ComparisonScore(cv::mean(diff).val[0]);
		}
};

typedef BioFW::Template<FaceFeatureVector> FaceTemplate;
typedef BioFW::TemplateCreator<FaceTemplate> FaceTemplateCreator;

typedef BioFW::Comparator<FaceFeatureVector, FaceTemplate> FaceComparator;

class FacePersTemplate: public BioFW::MemoryPersistentTemplate {
	public:
		FacePersTemplate() {

		}
		typedef std::shared_ptr<FacePersTemplate> Ptr;
		FaceTemplate faceTemplate;
};

class FaceTempalteSerializer: public BioFW::ITemplateSerializer<FaceTemplate, FacePersTemplate> {
	public:

		virtual PersistentTemplatePtr serialize(const Template & from) {

			FacePersTemplate::Ptr fpt(new FacePersTemplate());
			fpt->faceTemplate = from;
			return fpt;
		}
		virtual Template deserialize(PersistentTemplatePtr fptr) {
			return fptr->faceTemplate;
		}
};

typedef BioFW::ProcessingBlock<FaceSample, FaceTemplate, FacePersTemplate> Block;
typedef BioFW::IProcessingBlock<FaceSample> IBlock;

typedef IBlock::Ptr IBlockPtr;
typedef std::deque <IBlockPtr> Blocks;

#endif /* STRUCTURES_H_ */
