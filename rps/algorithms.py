# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:53:18 2021

@author: hoangdh
@email: snacky0907@gmail.com

"""

def game(p1,p2):
    if p1 == p2:
        return "Draw"
    else:
        if p1 == "Paper":
            if p2 == "Rock":
                return "P1 Win, P2 Lose"
            else:
                return "P1 Lose, P2 Win"
        elif p1 == "Rock":
            if p2 == "Paper":
                return "P2 Win, P1 Lose"
            else:
                return "P2 Lose, P1 Win"
        else:
            if p2 == "Paper":
                return "P1 Win, P2 Lose"
            else:
                return "P1 Lose, P2 Win"