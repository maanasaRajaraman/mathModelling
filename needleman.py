# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 14:38:14 2025

@author: maana
"""

def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    n, m = len(seq1), len(seq2) 
    score = [[0]*(m+1) for _ in range(n+1)]
 
    for i in range(1, n+1):
        score[i][0] = score[i-1][0] + gap
    for j in range(1, m+1):
        score[0][j] = score[0][j-1] + gap
 
    for i in range(1, n+1):
        for j in range(1, m+1):
            diag = score[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            up = score[i-1][j] + gap
            left = score[i][j-1] + gap
            score[i][j] = max(diag, up, left)
 
    align1, align2 = '', ''
    i, j = n, m
    while i > 0 or j > 0:
        current = score[i][j]
        if i > 0 and j > 0 and current == score[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch):
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            i, j = i-1, j-1
        elif i > 0 and current == score[i-1][j] + gap:
            align1 = seq1[i-1] + align1
            align2 = '-' + align2
            i -= 1
        else:
            align1 = '-' + align1
            align2 = seq2[j-1] + align2
            j -= 1

    return align1, align2, score[n][m]


def pretty_print_alignment(a1, a2):
    middle = []
    for x, y in zip(a1, a2):
        if x == y:
            middle.append('|')
        elif x == '-' or y == '-':
            middle.append(' ')
        else:
            middle.append(':')
    print("===  Needleman-Wunsch (Global)  ===")
    print(a1)
    print(''.join(middle))
    print(a2)


# Example
# s1 = 'ATCGT'
# s2 = 'TGGTG'
s1 = "GATTACA"
s2 = "GCATGCU"

a1, a2, sc = needleman_wunsch(s1, s2, match=1, mismatch=-1, gap=-1)
pretty_print_alignment(a1, a2)
print('Score:', sc)