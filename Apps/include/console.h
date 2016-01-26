/*
 * console.h
 *
 *  Created on: 25.1.2016
 *      Author: JV
 */

#ifndef CONSOLE_H_
#define CONSOLE_H_
#include "structures.h"
#include "BioFW/Utils/Database.h"


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



#endif /* CONSOLE_H_ */
