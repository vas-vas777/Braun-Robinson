#include <algorithm>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <fstream>
#include "matplotlibcpp.h"
#include "NumCpp-master\include\NumCpp\Linalg.hpp"
#include "NumCpp-master\include\NumCpp\NdArray.hpp"

namespace plt = matplotlibcpp;
std::vector<std::vector<double>> matrix_C = { {0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0} }; //ìàòðèöà Ñ
std::vector<std::vector<double>> reverse_matrix_C = { {0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0} }; //îáðàòíàÿ ìàòðèöà ê ìàòðèöå Ñ
std::vector<std::vector<double>> optimal_strategy_x = { { 0.0,0.0,0.0 } }; //îïòèìàëüíàÿ ñòðàòåãèÿ õ*
std::vector<std::vector<double>> optimal_strategy_y = { { 0.0}, {0.0}, {0.0} }; //îïòèìàëüíàÿ ñòðàòåãèÿ ó*
std::vector<double> graph_of_errors;
double e_error = 1.0; //âåëè÷èíà îøèáêè
double cost_of_game_v = 0.0; //öåíà èãðû

double det_matrix_C() //âû÷èñëåíèå îïðåäåëèòåëÿ ìàòðèöû 3Õ3
{
    double det = 0.0; //îïðåäåëèòåëü 
    det += matrix_C[0][0] * matrix_C[1][1] * matrix_C[2][2];
    det += matrix_C[0][1] * matrix_C[1][2] * matrix_C[2][0];
    det += matrix_C[0][2] * matrix_C[1][0] * matrix_C[2][1];
    det -= matrix_C[0][2] * matrix_C[1][1] * matrix_C[2][0];
    det -= matrix_C[0][0] * matrix_C[1][2] * matrix_C[2][1];
    det -= matrix_C[0][1] * matrix_C[1][0] * matrix_C[2][2];
    return det;
}

void alg_dopolnenie() //âû÷èñëåíèå àëãåáðàè÷åñêèõ äîïîëíåíèé
{
    reverse_matrix_C[0][0] = matrix_C[1][1] * matrix_C[2][2] - matrix_C[2][1] * matrix_C[1][2];
    reverse_matrix_C[0][1] = -(matrix_C[1][0] * matrix_C[2][2] - matrix_C[2][0] * matrix_C[1][2]);
    reverse_matrix_C[0][2] = matrix_C[1][0] * matrix_C[2][1] - matrix_C[2][0] * matrix_C[1][1];
    reverse_matrix_C[1][0] = -(matrix_C[0][1] * matrix_C[2][2] - matrix_C[2][1] * matrix_C[0][2]);
    reverse_matrix_C[1][1] = matrix_C[0][0] * matrix_C[2][2] - matrix_C[2][0] * matrix_C[0][2];
    reverse_matrix_C[1][2] = -(matrix_C[0][0] * matrix_C[2][1] - matrix_C[2][0] * matrix_C[0][1]);
    reverse_matrix_C[2][0] = matrix_C[0][1] * matrix_C[1][2] - matrix_C[1][1] * matrix_C[0][2];
    reverse_matrix_C[2][1] = -(matrix_C[0][0] * matrix_C[1][2] - matrix_C[1][0] * matrix_C[0][2]);
    reverse_matrix_C[2][2] = matrix_C[0][0] * matrix_C[1][1] - matrix_C[1][0] * matrix_C[0][1];
}

void transport_matrix_of_alg_dop() //òðàíñïîðòèðîâàíèå ìàòðèöû, ñîñòàâëåííîé èç àëãåáðàè÷åñêèõ äîïîëíåíèé 
{

    double temp = 0.0;
    temp = reverse_matrix_C[0][1];
    reverse_matrix_C[0][1] = reverse_matrix_C[1][0];
    reverse_matrix_C[1][0] = temp;
    temp = reverse_matrix_C[0][2];
    reverse_matrix_C[0][2] = reverse_matrix_C[2][0];
    reverse_matrix_C[2][0] = temp;
    temp = reverse_matrix_C[1][2];
    reverse_matrix_C[1][2] = reverse_matrix_C[2][1];
    reverse_matrix_C[2][1] = temp;

    for (size_t index_row = 0; index_row < 3; ++index_row)
    {
        for (size_t index_column = 0; index_column < 3; ++index_column)
        {
            reverse_matrix_C[index_row][index_column] *= 1.0 / det_matrix_C();
        }
    }
}


void enter_values_of_matrix_C(int argc, char *argv[]) //ââîä ìàòðèöû è ïðîâåðêà íà ââîä
{
    std::string letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUWXYZàáâãäå¸æçèéêëìíîïðñòóôçö÷øùúûüýþÿÀÁÂÃÄÅ¨ÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß";
    size_t count_argv = 1;
    if (argc <= 9)
    {
        std::cout << "Íå âñå ÷èñëà ââåäåíû" << std::endl;
        exit(1);
    }
    if (argc > 10)
    {
        std::cout << "Ñëèøêîì ìíîãî äàííûõ. Ó÷èòûâàþòñÿ òîëüêî  ïåðâûå 9 ÷èñåë" << std::endl;
    }
    std::string str_argv = "";
    for (size_t index = 1; index < argc; ++index)
    {
        if (index < 10)
        {
            str_argv += std::string(argv[index]);
        }
        else
            break;
    }
    if (str_argv.find_first_of(letters) != std::string::npos)
    {
        std::cout << "Ââåäåíà áóêâà" << std::endl;
        exit(1);
    }
    for (size_t index_row = 0; index_row < 3; ++index_row)
    {
        for (size_t index_column = 0; index_column < 3; ++index_column)
        {
            if (count_argv <= 9)
            {
                matrix_C[index_row][index_column] = atoi(argv[count_argv]);
                count_argv++;
            }
            else
                break;
        }
        std::cout << std::endl;
    }
}


void analitic_method() //ðàñ÷¸ò àíàëèòè÷åñêèì ìåòîäîè
{
    //ðàñ÷¸ò îïòèìàëüíîé ñòàðòåãèè õ*= (u*C^(-1))/(u*C^(-1)*u^T), ãäå u=(1,1,1), u^T-òðàíñïîíèðîâàííûé âåêòîð u
    // temp - çíà÷åíèå u*C^(-1)*u^T
    //ðàñ÷¸ò îïòèìàëüíîé ñòàðòåãèè y*= (C^(-1)*u^T)/(u*C^(-1)*u^T), ãäå u=(1,1,1), u^T-òðàíñïîíèðîâàííûé âåêòîð u
    //v=1/(u*C^(-1)*u^T)

    double temp = 0.0;
    //u*C^(-1)
    optimal_strategy_x[0][0] = reverse_matrix_C[0][0] + reverse_matrix_C[1][0] + reverse_matrix_C[2][0];
    optimal_strategy_x[0][1] = reverse_matrix_C[0][1] + reverse_matrix_C[1][1] + reverse_matrix_C[2][1];
    optimal_strategy_x[0][2] = reverse_matrix_C[0][2] + reverse_matrix_C[1][2] + reverse_matrix_C[2][2];
    temp = optimal_strategy_x[0][0] + optimal_strategy_x[0][1] + optimal_strategy_x[0][2];//u*C^(-1)*u^T
    for (size_t index_column = 0; index_column < 3; ++index_column)
    {
        optimal_strategy_x[0][index_column] *= 1.0 / temp; //âåêòîð x*
    }
    //C^(-1)*u^T
    optimal_strategy_y[0][0] = reverse_matrix_C[0][0] + reverse_matrix_C[0][1] + reverse_matrix_C[0][2];
    optimal_strategy_y[1][0] = reverse_matrix_C[1][0] + reverse_matrix_C[1][1] + reverse_matrix_C[1][2];
    optimal_strategy_y[2][0] = reverse_matrix_C[2][0] + reverse_matrix_C[2][1] + reverse_matrix_C[2][2];
    for (size_t index_row = 0; index_row < 3; ++index_row)
    {
        optimal_strategy_y[index_row][0] *= 1.0 / temp; //âåêòîð y*
    }
    //öåíà èãðû
    cost_of_game_v = 1.0 / temp;
}

void braun_robin() //àëãîðèòì Áðàóíà-Ðîáèíñîíà
{
    std::fstream result_file("C:\\ÌÃÒÓ\\ÒåîðèÿÈãð\\Lab1-Braun-Robins\\Lab1-Braun-Robins\\table.csv", std::ios_base::out | std::ios_base::trunc);
    
    std::vector<int> wins_of_A = { 0,0,0 }; //âûèãðàø èãðîêà À
    std::vector<int> loses_of_B = { 0,0,0 };//ïðîèãðûø èãðîêà Â

    std::vector<double> high_cost_of_game; // ñðåäíèè çíà÷åíèÿ âåðõíåé öåíû èãðû
    std::vector<double>::iterator iterator_of_high_cost_of_game;
    std::vector<double> low_cost_of_game; // ñðåäíèè çíà÷åíèÿ íèæíåé öåíû èãðû
    std::vector<double>::iterator iterator_of_low_cost_of_game;
    double count = 1.0; //êîëè÷åñòâî øàãîâ 
    int pos_min_element_of_high_cost_of_game = 0; //ïîçèöèÿ ìèíèìàëüíîãî ýëåìåíòà â ìàññèâå ñðåäíèõ çíà÷åíèé âåðõíåé öåíû èãðû
    int pos_max_element_of_low_cost_of_game = 0; //ïîçèöèÿ ìàêñèìàëüíîãî ýëåìåíòà â ìàññèâå ñðåäíèõ çíà÷åíèé âåðõíåé öåíû èãðû
    //êîëè÷åñòâî èñïîëüçîâàííûõ ñòðàòåãèé èãðîêàìè À è Â
    int count_of_x1 = 1;
    int count_of_x2 = 0;
    int count_of_x3 = 0;
    int count_of_y1 = 1;
    int count_of_y2 = 0;
    int count_of_y3 = 0;
    
    //âûáîð ñòðàòåãèè õ1 èãðîêîì À íà øàãå 1
    wins_of_A[0] = matrix_C[0][0];
    wins_of_A[1] = matrix_C[1][0];
    wins_of_A[2] = matrix_C[2][0];
    //âûáîð ñòðàòåãèè ó1 èãðîêîì Â íà øàãå 1
    loses_of_B[0] = matrix_C[0][0];
    loses_of_B[1] = matrix_C[0][1];
    loses_of_B[2] = matrix_C[0][2];

    //íàõîæäåíèå ìàêñèìàëüíîãî âûèãðàøà ó èãðîêà À
    std::vector<int>::iterator max_result_of_A;
    max_result_of_A = std::max_element(wins_of_A.begin(), wins_of_A.end());
    int pos_max_win_of_A = std::distance(wins_of_A.begin(), max_result_of_A);
    //íàõîæäåíèå ìèíèìàëüíîãî ïðîèãðûøà ó èãðîêà Â
    std::vector<int>::iterator min_result_of_B;
    min_result_of_B = std::min_element(loses_of_B.begin(), loses_of_B.end());
    int pos_min_lose_of_B = std::distance(loses_of_B.begin(), min_result_of_B);
    //ïîäñ÷¸ò ñðåäíåãî çíà÷åíèÿ âåðõíåé öåíû èãðû 
    high_cost_of_game.push_back(wins_of_A.at(pos_max_win_of_A) * 1.0 / count);
    //ïîäñ÷¸ò ñðåäíåãî çíà÷åíèÿ íèæíåé öåíû èãðû
    low_cost_of_game.push_back(loses_of_B.at(pos_min_lose_of_B) * 1.0 / count);
    //âû÷èñëåíèå çíà÷åíèÿ îøèáêè
    e_error = high_cost_of_game.at(0) - low_cost_of_game.at(0);
    graph_of_errors.push_back(e_error);

    if (e_error > 0.1);
    result_file << "k" << ";" << "âûáîð èãðîêà" << ";" << "âûèãðûø èãðîêà À" << ";"<<";" <<";"<< "ïðîèãðûø èãðîêà Â" << ";" <<";"<<";"<< "1 / k * v[k] - âåðõ" <<";"<< "1/k*v[k]-íèæ|" << ";" << "e-ïîãðåøíîñòü" << "\n";
    result_file << " " << ";" << "A|B" << ";" << "x1" << ";" << "x2" << ";" << "x3" << ";" << "y1" << ";" << "y2" << ";" << "y3" << "\n";

    do
    {

        if (count == 1)
        {
            result_file<<count<<";"<<"x1 y1"<<";"<<wins_of_A[0] << ";" << wins_of_A[1] << ";" << wins_of_A[2]<<";"<< loses_of_B[0] << ";" << loses_of_B[1] << ";" << loses_of_B[2] <<";"<< floor(high_cost_of_game.back() * 100) / 100<<";"<< floor(low_cost_of_game.back() * 100) / 100<<";"<< floor(e_error * 100) / 100 << std::endl;
        }
        count++;
        result_file << count << ";";
        if (0 == pos_max_win_of_A) //åñëè ìàêñèìàëüíîå çíà÷åíèå áóäåò ïðè âûáîðå ñòðàòåãèè õ1
        {
            loses_of_B[0] += matrix_C[0][0];
            loses_of_B[1] += matrix_C[0][1];
            loses_of_B[2] += matrix_C[0][2];
            count_of_x1++;
            result_file << "x1 ";
        }
        else if (1 == pos_max_win_of_A)//åñëè ìàêñèìàëüíîå çíà÷åíèå áóäåò ïðè âûáîðå ñòðàòåãèè õ2
        {
            loses_of_B[0] += matrix_C[1][0];
            loses_of_B[1] += matrix_C[1][1];
            loses_of_B[2] += matrix_C[1][2];
            count_of_x2++;
            result_file << "x2 ";
        }
        else if (2 == pos_max_win_of_A)//åñëè ìàêñèìàëüíîå çíà÷åíèå áóäåò ïðè âûáîðå ñòðàòåãèè õ3
        {
            loses_of_B[0] += matrix_C[2][0];
            loses_of_B[1] += matrix_C[2][1];
            loses_of_B[2] += matrix_C[2][2];
            count_of_x3++;
            result_file << "x3 ";
        }

        if (0 == pos_min_lose_of_B)//åñëè ìèíèìàëüíûé ïðîèãðûø áóäåò ïðè âûáîðå ñòðàòåãèè ó1
        {
            wins_of_A[0] += matrix_C[0][0];
            wins_of_A[1] += matrix_C[1][0];
            wins_of_A[2] += matrix_C[2][0];
            count_of_y1++;
            result_file << "y1" << ";";
        }
        else if (1 == pos_min_lose_of_B)//åñëè ìèíèìàëüíûé ïðîèãðûø áóäåò ïðè âûáîðå ñòðàòåãèè ó2
        {
            wins_of_A[0] += matrix_C[0][1];
            wins_of_A[1] += matrix_C[1][1];
            wins_of_A[2] += matrix_C[2][1];
            count_of_y2++;
            result_file << "y2" << ";";
        }
        else if (2 == pos_min_lose_of_B)//åñëè ìèíèìàëüíûé ïðîèãðûø áóäåò ïðè âûáîðå ñòðàòåãèè ó3
        {
            wins_of_A[0] += matrix_C[0][2];
            wins_of_A[1] += matrix_C[1][2];
            wins_of_A[2] += matrix_C[2][2];
            count_of_y3++;
            result_file << "y3" << ";";
        }
      
        result_file <<wins_of_A[0] << ";" << wins_of_A[1] << ";" << wins_of_A[2] << ";" << loses_of_B[0] << ";" << loses_of_B[1] << ";" << loses_of_B[2] << ";" ;
        //íàõîæäåíèå ìàêñèìàëüíîãî âûèãðàøà ó èãðîêà À
        max_result_of_A = std::max_element(wins_of_A.begin(), wins_of_A.end());

        pos_max_win_of_A = std::distance(wins_of_A.begin(), max_result_of_A);
        //íàõîæäåíèå ìèíèìàëüíîãî ïðîèãðûøà ó èãðîêà Â
        min_result_of_B = std::min_element(loses_of_B.begin(), loses_of_B.end());

        pos_min_lose_of_B = std::distance(loses_of_B.begin(), min_result_of_B);
          //ïîäñ÷¸ò ñðåäíåãî çíà÷åíèÿ âåðõíåé öåíû èãðû 
        high_cost_of_game.push_back(wins_of_A.at(pos_max_win_of_A) * 1.0 / count);
        //ïîäñ÷¸ò ñðåäíåãî çíà÷åíèÿ íèæíåé öåíû èãðû
        low_cost_of_game.push_back(loses_of_B.at(pos_min_lose_of_B) * 1.0 / count);
        //ïîèñê ìèíèìàëüíîãî ýëåìåíòà ñðåäè çíà÷åíèé âåðõíåé öåíû èãðû
        iterator_of_high_cost_of_game = std::min_element(high_cost_of_game.begin(), high_cost_of_game.end());
        pos_min_element_of_high_cost_of_game = std::distance(high_cost_of_game.begin(), iterator_of_high_cost_of_game);
        //ïîèñê ìàêñèìàëüíîãî ýëåìåíòà ñðåäè çíà÷åíèé âåðõíåé öåíû èãðû
        iterator_of_low_cost_of_game = std::max_element(low_cost_of_game.begin(), low_cost_of_game.end());
        pos_max_element_of_low_cost_of_game = std::distance(low_cost_of_game.begin(), iterator_of_low_cost_of_game);

        //âû÷èñëåíèå âåëè÷èíû îøèáêè
        e_error = high_cost_of_game.at(pos_min_element_of_high_cost_of_game) - low_cost_of_game.at(pos_max_element_of_low_cost_of_game);
        result_file <<  floor(high_cost_of_game.back() * 100) / 100 << ";" << floor(low_cost_of_game.back() * 100) / 100 << ";" << floor(e_error * 100) / 100 << std::endl;
        graph_of_errors.push_back(e_error);

    } while (e_error > 0.1); //åñëè îèøáêà ìåíüøå 0.1, òî îñòàíàâëèâàåì öèêë
    //âûâîä ðåçóëüòàòîâ
    std::vector<double>::iterator max_result_of_cost;
    max_result_of_cost = std::max_element(high_cost_of_game.begin(), high_cost_of_game.end());
    int pos_max_cost_of_game = std::distance(high_cost_of_game.begin(), max_result_of_cost);

    std::vector<double>::iterator min_result_of_cost;
    min_result_of_cost = std::min_element(low_cost_of_game.begin(), low_cost_of_game.end());
    int pos_min_cost_of_game = std::distance(low_cost_of_game.begin(), min_result_of_cost);
    std::cout << *std::min_element(high_cost_of_game.begin(), high_cost_of_game.end()) << " " << *std::max_element(low_cost_of_game.begin(), low_cost_of_game.end()) << std::endl;
    cost_of_game_v = high_cost_of_game.at(pos_max_cost_of_game) - low_cost_of_game.at(pos_min_cost_of_game);

    std::cout << "Çíà÷åíèå îøèáêè è êîëè÷åñòâî øàãîâ" << std::endl;
    std::cout << "e=" << e_error << std::endl;
    std::cout << "k=" << count << std::endl;
    std::cout << "Âûâîä ïðèáëèæåííûõ çíà÷åíèé ñìåøàííûõ ñòðàòåãèé" << std::endl;
    std::cout << "x*[" << count << "]=(" <<count_of_x1<<"/"<<count<<"="<< double(count_of_x1 / count) << "," << count_of_x2 << "/" << count << "=" << double(count_of_x2 / count) << "," << count_of_x3 << "/" << count << "=" << double(count_of_x3 / count) << ")" << std::endl;
    std::cout << "y*[" << count << "]=(" << count_of_y1 << "/" << count << "=" << double(count_of_y1 / count) << "," << count_of_y2 << "/" << count << "=" << double(count_of_y2 / count) << "," << count_of_y3 << "/" << count << "=" << double(count_of_y3 / count) << ")" << std::endl;
    std::cout << "v=" << cost_of_game_v << std::endl;
    result_file.close();
}

void print_analitic_solve() //ïå÷àòü àíàëèòè÷åñêîãî ðåùåíèÿ
{
    std::cout << "x*=(";
    for (size_t index_column = 0; index_column < 3; ++index_column)
    {
        std::cout << optimal_strategy_x[0][index_column] << " ";  //âåêòîð x*
    }
    std::cout << ")";
    std::cout << std::endl;
    std::cout << "y*=(";
    for (size_t index_row = 0; index_row < 3; ++index_row)
    {
        std::cout << optimal_strategy_y[index_row][0] << " "; //âåêòîð y*
    }
    std::cout << ")";
    std::cout << std::endl;
    std::cout << "v=" << cost_of_game_v << std::endl;
  
    plt::plot(graph_of_errors);//ïîñòðîåíèå ãðàôèêà
    plt::xlabel("Counts");
    plt::ylabel("e-error");
    plt::show();

}
