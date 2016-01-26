/*
 * primaryBanks.h
 *
 *  Created on: 12.3.2014
 *      Author: JV
 */

#ifndef PRIMARYBANKS_H_
#define PRIMARYBANKS_H_

#include "structures.h"
#include <vector>
#include "BioFW/FeatureVector/ChainedFeatureExtractor.h"

#include <map>

#include <functional>

#include <BioFW/FeatureVector/SpatialHistogram/CLBPDistHistogramExtractor.h>
#include <BioFW/FeatureVector/SpatialHistogram/CombinedLBPHHistogramExtractor.h>
#include <BioFW/FeatureVector/SpatialHistogram/LBPHistogramExtractor.h>
#include <BioFW/FeatureVector/SpatialHistogram/LTPHistogramExtractor.h>
#include <BioFW/FeatureVector/SpatialHistogram/WLDHistogramExtractor.h>

template <class FExtractor>
typename FaceFeatureExtractorWrapper<FExtractor>::Ptr wrap(FaceSample::ImageType t){
		return typename FaceFeatureExtractorWrapper<FExtractor>::Ptr(
				new FaceFeatureExtractorWrapper<FExtractor>(
						std::make_shared<FExtractor>(),
						t
				)
		);
}

template <class FExtractor>
typename FaceFeatureExtractorWrapper<FExtractor>::Ptr wrap(FaceSample::ImageType t, typename FExtractor::Ptr fe){
		return typename FaceFeatureExtractorWrapper<FExtractor>::Ptr(
				new FaceFeatureExtractorWrapper<FExtractor>(
						fe,
						t
				)
		);
}

typedef BioFW::IFeatureVectorExtractor<BioFW::MatFV, BioFW::MatFV> BasicExtractor;
typedef BasicExtractor::Ptr BasicExtractorPtr;


typedef std::function<BasicExtractor::Ptr ()> BasicExtractorFactory;

typedef std::map<std::string, BasicExtractorFactory> BasicExtractorFactories;
typedef std::map<std::string, BasicExtractorPtr> BasicExtractos;

typedef std::function<BasicExtractos ()> BasicExtractorsFactory;


BasicExtractorsFactory basicGabors(){
	return [] () -> BasicExtractos {
		BasicExtractos extractors;
		for (int freq = 5; freq <= 5; freq+= 1){
			for (int o = 0; o <= 3; o+=1){
				std::stringstream s;
				s << "Gabor(" << freq << "," << o <<")";

				extractors.insert(std::make_pair(
						s.str(),
						BioFW::GaborFeatureExtractor::Ptr(new BioFW::GaborFeatureExtractor(freq, o, 1))
				));
			}
		}
		return extractors;
	};
}


typedef BioFW::ChainedFeatureExtractor<BasicExtractor, BasicExtractor> Chained;

BasicExtractos wrapByPCA(BasicExtractos & oldextractors){
	BasicExtractos extractors;

	for (auto o : oldextractors){
		std::stringstream s;
		s << o.first << "+" << "PCA";
		Chained::Ptr chained(
				new Chained(
					o.second,
					BioFW::EigenfacesExtractor::Ptr(new BioFW::EigenfacesExtractor(0.995))
				)
			);
		extractors.insert(std::make_pair(s.str(), chained));
	}
	return extractors;
}

BasicExtractos wrapByZScore(BasicExtractos & oldextractors){
	BasicExtractos extractors;

	for (auto o : oldextractors){
		std::stringstream s;
		s << o.first << "+" << "Z";
		Chained::Ptr chained(
				new Chained(
					o.second,
					BioFW::ZScoreExtractor::Ptr(new BioFW::ZScoreExtractor())
				)
			);
		extractors.insert(std::make_pair(s.str(), chained));
	}
	return extractors;
}

BasicExtractorsFactory advancedGabors(){
	return [] () -> BasicExtractos {
		BasicExtractorsFactory bg = basicGabors();
		BasicExtractos ext = bg();
		return wrapByPCA(ext);
	};
}

BasicExtractorsFactory extraAdvancedGabors(){
	return [] () -> BasicExtractos {
		BasicExtractorsFactory bg = advancedGabors();
		BasicExtractos ext = bg();
		return wrapByZScore(ext);
	};
}

/*

*/
IBlock::Ptr makeBlock(std::string name, FaceSample::ImageType t, BasicExtractor::Ptr be, BioFW::IFeatureVectorComparator<BioFW::MatFV>::Ptr c){
	return IBlock::Ptr(
		new Block(
				name,
				wrap<BasicExtractor>(t, be),
				std::make_shared < FaceComparator > (c),
				std::make_shared<FaceTemplateCreator>(),
				std::make_shared<FaceTempalteSerializer>()
		)
	);
}

Blocks makeBlocks(std::string name, BasicExtractorFactory beFactory, BioFW::IFeatureVectorComparator<BioFW::MatFV>::Ptr c){


	Blocks b;

	std::map < FaceSample::ImageType, std::string> images;
	//images[FaceSample::Project3D] = "P";
	images[FaceSample::Project3D_Global] = "PG";
	//images[FaceSample::Project3D_Local] = "PL";

	//images[FaceSample::Warp2D] = "W";
	images[FaceSample::Warp2D_Global] = "WG";
	//images[FaceSample::Warp2D_Local] = "WL";

	for (auto i : images){
		std::string n = i.second + "+" + name;
		b.push_back(makeBlock(n, i.first, beFactory(), c));
	}
	return b;
}

Blocks makeBlocks(BasicExtractorsFactory beFactory, BioFW::IFeatureVectorComparator<BioFW::MatFV>::Ptr c){
	Blocks b;

	std::map < FaceSample::ImageType, std::string> images;
	//images[FaceSample::Project3D] = "P";
	images[FaceSample::Project3D_Global] = "PG";
	//images[FaceSample::Project3D_Local] = "PL";

	//images[FaceSample::Warp2D] = "W";
	images[FaceSample::Warp2D_Global] = "WG";
	//images[FaceSample::Warp2D_Local] = "WL";

	for (auto i : images){
		BasicExtractos e = beFactory();
		for (auto fb : e){
			std::string n = i.second + "+" + fb.first;
			b.push_back(makeBlock(n, i.first, fb.second, c));
		}
	}
	return b;
}


Blocks createBlocks(){
	typedef BioFW::EigenfacesExtractor EigenFE;

	Blocks blocks;
	{
		typedef BioFW::WLDHistogramExtractor/*<BioFW::MatSample, FaceFeatureVector>*/ WLDFE;
		{
			Blocks b = makeBlocks(
			"wld",
						[] () -> BasicExtractorPtr { return std::make_shared<WLDFE>(); },
						std::make_shared<WLDFE::DefaultComparator>()
				);
				blocks.insert(blocks.end(), b.begin(), b.end());
		}
		{
			Blocks b = makeBlocks(
			"wld12",
						[] () -> BasicExtractorPtr { return WLDFE::Ptr(new WLDFE(12,12)); },
						std::make_shared<WLDFE::DefaultComparator>()
				);
				blocks.insert(blocks.end(), b.begin(), b.end());
		}
		{
			Blocks b = makeBlocks(
			"wld16",
						[] () -> BasicExtractorPtr { return WLDFE::Ptr(new WLDFE(16,16)); },
						std::make_shared<WLDFE::DefaultComparator>()
				);
				blocks.insert(blocks.end(), b.begin(), b.end());
		}
		/*
		{
			Blocks b = makeBlocks(
					"wld+PCA",
					[] () -> BasicExtractorPtr {
						return Chained::Ptr(
							new Chained(
								std::make_shared<WLDFE>(),
								BioFW::EigenfacesExtractor::Ptr(new BioFW::EigenfacesExtractor(0.95))
							)
						);  },
					std::make_shared<EigenFE::DefaultComparator>()
			);
			blocks.insert(blocks.end(), b.begin(), b.end());
		}*/

	}


	{
			typedef BioFW::LBPHistogramExtractor LBPFE;
			{
				Blocks b = makeBlocks(
				"lbp",
							[] () -> BasicExtractorPtr { return std::make_shared<LBPFE>(); },
							std::make_shared<LBPFE::DefaultComparator>()
					);
					blocks.insert(blocks.end(), b.begin(), b.end());
			}
	}
/*
	{
			typedef BioFW::CLBPDistHistogramExtractor cLBPDistFE;
			{
				Blocks b = makeBlocks(
				"cLBPDist",
							[] () -> BasicExtractorPtr { return std::make_shared<cLBPDistFE>(); },
							std::make_shared<cLBPDistFE::DefaultComparator>()
					);
					blocks.insert(blocks.end(), b.begin(), b.end());
			}
	}

	{
			typedef BioFW::LTPHistogramExtractor LTPFE;
			{
				Blocks b = makeBlocks(
				"ltp",
							[] () -> BasicExtractorPtr { return std::make_shared<LTPFE>(); },
							std::make_shared<LTPFE::DefaultComparator>()
					);
					blocks.insert(blocks.end(), b.begin(), b.end());
			}
	}*/

	{
			typedef BioFW::CombinedLBPHHistogramExtractor<BioFW::MatSample, FaceFeatureVector> CombLBPHFE;
			Blocks b = makeBlocks(
					"CombinedLBPH",
					[] () -> BasicExtractorPtr { return std::make_shared<CombLBPHFE>(); },
					std::make_shared<CombLBPHFE::DefaultComparator>()
			);
			blocks.insert(blocks.end(), b.begin(), b.end());
		}

/*
	{
		BasicExtractorsFactory f = basicGabors();
		Blocks b = makeBlocks(
					f,
					std::make_shared<BioFW::NormComparator<cv::NORM_L1, BioFW::MatFV, BioFW::MatFV> >()
		);
		blocks.insert(blocks.end(), b.begin(), b.end());

	}*/

	{

		Blocks b = makeBlocks(
				"PCAr95",
				[] () -> BasicExtractorPtr { return EigenFE::Ptr(new EigenFE(0.95)); },
				std::make_shared<EigenFE::DefaultComparator>()
		);
		blocks.insert(blocks.end(), b.begin(), b.end());
	}
/*
	{

		Blocks b = makeBlocks(
				"LDA",
				[] () -> BasicExtractorPtr { return BioFW::FisherfacesExtractor::Ptr(new  BioFW::FisherfacesExtractor()); },
				std::make_shared<BioFW::FisherfacesExtractor::DefaultComparator>()
		);
		blocks.insert(blocks.end(), b.begin(), b.end());
	}
*/
/*
	{
		BasicExtractorsFactory f = advancedGabors();
		Blocks b = makeBlocks(
					f,
					std::make_shared<EigenFE::DefaultComparator>()
		);
		blocks.insert(blocks.end(), b.begin(), b.end());
	}
*/
/*
	{
		BasicExtractorsFactory f = extraAdvancedGabors();
		Blocks b = makeBlocks(
					f,
					std::make_shared<BioFW::CosineComparator>()
		);
		blocks.insert(blocks.end(), b.begin(), b.end());
	}*/

	return blocks;
}

#endif /* PRIMARYBANKS_H_ */
