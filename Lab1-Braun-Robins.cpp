#include<fstream>
#include <iostream>
#include <locale.h>

#include "Braun-Robinson.h"


int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "RUS");
    enter_values_of_matrix_C(argc, argv);
    alg_dopolnenie();
    transport_matrix_of_alg_dop();
    analitic_method();
    braun_robin();
    std::cout << "Вывод значений смешанных стратегий, полученных с помощью аналитического метода" << std::endl;
    print_analitic_solve();
  
}

