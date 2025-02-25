# Authors: marcusm117
# License: Apache 2.0


# Standard Library Modules
import json

# External Modules
import pandas as pd


# EXAMPLE USAGE: python construct_AIME_2024_data_splits.py
def main():
    # load the ArtofProblemSolving dataset, which is downloaded from
    # https://www.kaggle.com/datasets/alexryzhkov/amio-parsed-art-of-problem-solving-website/data
    df = pd.read_csv("parsed_ArtOfProblemSolving.csv").drop(columns=["problem_id"])

    # select all the rows with "link" column not containing "2024_AIME" as the train split
    df_train = df[~df["link"].str.contains("2024_AIME")]

    # drop the rows with duplicate "problem" and "solution" columns
    # we keep multiple different solutions to the same problem
    df_train = df_train.drop_duplicates(subset=["problem", "solution"])

    # convert to a jsonl file, where multiple solutions to the same problem are stored in a list
    train_data = {}
    for _, row in df_train.iterrows():
        link = row["link"]
        solution = row["solution"]
        # remove all the options from the problem
        problem = row["problem"].split("$\\textbf{(A")[0].strip()
        # if the letter is missing, save it as an empty string
        letter = row["letter"] if pd.notna(row["letter"]) else ""
        # remove trailing decimal points and trailing commas
        answer = row["answer"].strip().rstrip(".").rstrip(",")
        answer_parts = answer.split(",")
        # if the answer contains a decimal point, save it as float
        if "." in answer:
            answer = float(answer)
        # if the answer contains no comma, save it as an integer
        elif len(answer_parts) == 1:
            try:
                answer = int(answer)
            except ValueError:
                print(f"ValueError: {answer} for {row['link']}")
                return
        # if the answer contains commas, all parts are digits, first part is <= 3 digits, and all other parts are 3 digits
        elif len(answer_parts[0]) <= 3 and all(len(part) == 3 for part in answer_parts[1:]) and all(part.isdigit() for part in answer_parts):
            answer = int(answer.replace(",", ""))
        # otherwise, save the answer as it is

        # fix or skip some partial or wrong solutions, skip some abiguous problems
        if link == "https://artofproblemsolving.com/wiki/index.php/2004_AMC_8_Problems/Problem_17" and answer == 4:
            solution = r"""Like in solution 1 and solution 2, assign one pencil to each of the three friends so that you have $3$ left. In partitioning the remaining $3$ pencils into $3$ distinct groups use casework. Let the three friends be $a$, $b$, $c$ repectively.

$a + b + c = 3$,


Case $1:a=0$,

$b + c = 3$,

$b = 0,1,2,3$ ,

$c = 3,2,1,0$,

$\boxed{\textbf\ 4}$ solutions.


Case $2:a=1$,

$1 + b + c = 3$,

$b + c = 2$,

$b = 0,1,2$ ,

$c = 2,1,0$ ,

$\boxed{\textbf\ 3}$ solutions.


Case $3:a= 2$,

$2 + b + c = 3$,

$b + c = 1$,

$b = 0,1$,

$c = 1,0$,

$\boxed{\textbf\ 2}$ solutions.


Case $4:a = 3$,

$3 + b + c = 3$,

$b + c = 0$,

$b = 0$,

$c = 0$,

$\boxed{\textbf\ 1}$ solution.

Therefore there will be a total of $4+3+2+1=10$ solutions. $\boxed{10}$."""
            answer = 10
        elif link == "https://artofproblemsolving.com/wiki/index.php/1996_AHSME_Problems/Problem_17" and answer == 3:
            solution = r"""Since $\angle C = 90^\circ$, each of the three smaller angles is $30^\circ$, and $\triangle BEC$ and $\triangle CDF$ are both $30-60-90$ triangles.

[asy] pair A=origin, B=(10,0), C=(10,7), D=(0,7), E=(5,0), F=(0,2); draw(A--B--C--D--cycle, linewidth(0.8)); draw(E--C--F); dot(A^^B^^C^^D^^E^^F); label("$A$", A, dir((5, 3.5)--A)); label("$B$", B, dir((5, 3.5)--B)); label("$C$", C, dir((5, 3.5)--C)); label("$D$", D, dir((5, 3.5)--D)); label("$E$", E, dir((5, 3.5)--E)); label("$F$", F, dir((5, 3.5)--F)); label("$2$", (0,1), plain.E, fontsize(10)); label("$x$", (9,3.5), E, fontsize(10)); label("$x-2$", (0,5), plain.E, fontsize(10)); label("$y$", (5,7), N, fontsize(10)); label("$6$", (7.5,0), S, fontsize(10));[/asy]
Defining the variables as illustrated above, we have $x = 6\sqrt{3}$ from $\triangle BEC$

Then $x-2 = 6\sqrt{3} - 2$, and $y = \sqrt{3} (6 \sqrt{3} - 2) = 18 - 2\sqrt{3}$.

The area of the rectangle is thus $xy = 6\sqrt{3}(18 - 2\sqrt{3}) = 108\sqrt{3} - 36$.

Using the approximation $\sqrt{3} \approx 1.7$, we get an area of just under $147.6$, which is closest to answer $\boxed{150}$. (The actual area is actually greater, since $\sqrt{3} > 1.7$)."""
            answer = 150
        elif link == "https://artofproblemsolving.com/wiki/index.php/1965_AHSME_Problems/Problem_1" and answer == 1:
            solution = r"""Notice that $a^0=1, a>0$. So $2^0=1$. So $2x^2-7x+5=0$. Evaluating the discriminant, we see that it is equal to $7^2-4*2*5=9$. So this means that the equation has two real solutions. Therefore, the answer is $\boxed{2}$."""
            answer = 2
        elif link == "https://artofproblemsolving.com/wiki/index.php/2020_AMC_10A_Problems/Problem_19" and answer == 90:
            continue
        elif link == "https://artofproblemsolving.com/wiki/index.php/2017_AMC_10A_Problems/Problem_18" and answer == 59:
            solution = r"""We can solve this by using 'casework,' the cases being: Case 1: Amelia wins on her first turn. Case 2 Amelia wins on her second turn. and so on.

The probability of her winning on her first turn is $\dfrac13$. The probability of all the other cases is determined by the probability that Amelia and Blaine all lose until Amelia's turn on which she is supposed to win. So, the total probability of Amelia winning is:\[\dfrac{1}{3}+\left(\dfrac{2}{3}\cdot\dfrac{3}{5}\right)\cdot\dfrac{1}{3}+\left(\dfrac{2}{3}\cdot\dfrac{3}{5}\right)^2\cdot\dfrac{1}{3}+\cdots.\]Factoring out $\dfrac13$ we get a geometric series:\[\dfrac{1}{3}\left(1+\dfrac{2}{5}+\left(\dfrac{2}{5}\right)^2+\cdots\right) = \dfrac{1}{3}\cdot\dfrac{1}{3/5} = \boxed{\dfrac59}.\]
Extracting the desired result, we get $9-5 = \boxed{4}$."""
            answer = 4
        elif link == "https://artofproblemsolving.com/wiki/index.php/2012_AMC_12B_Problems/Problem_12":
            continue
        elif link == "https://artofproblemsolving.com/wiki/index.php/2023_AIME_II_Problems/Problem_7" and answer == 64:
            solution = r"""\[\text{First, we notice that a rectangle is made from two pairs of vertices 1/2 turn away from each other.}\]
\[\textit{Note: The image is }\frac{\textit{280}}{\textit{841}}\approx\frac{\textit{1}}{\textit{3}}\textit{ size.}\]
\[\text{For there to be no rectangles, there can be at most one same-colored pair for each color and the rest need to have one red and blue.}\]
\[\textit{\underline{Case 1: \textbf{No pairs}}}\]
\[\text{Each pair has two ways to color: One red or the other red. Thus, there are }2^6=\boxed{64}\text{ ways in this case.}\]
\[\textit{\underline{Case 2: \textbf{One red pair}}}\]
\[\text{The red pair has }\binom{6}{1}\text{ positions. All the rest still have two ways. Therefore, there are }\binom{6}{1}\cdot 2^5=\frac{6}{1}\dot 2^5=6\cdot 32=\boxed{192} \text{ ways in this case.}\]
\[\textit{\underline{Case 3: \textbf{One blue pair}}}\]
\[\text{This is the same as the one red pair case so there are still }\binom{6}{1}\cdot 2^5=6\cdot 2^5=6\cdot 32=\boxed{192}\text{ ways.}\]
\[\textit{\underline{Case 4: \textbf{One pair of each color}}}\]
\[\text{The red pair has }\binom{6}{1}\text{ positions. The blue pair has }\binom{5}{1}\text{ positions. All the rest still have two ways. Therefore, there are }\binom{6}{1}\cdot\binom{5}{1}\cdot 2^4=\frac{6\cdot 5=30}{1\cdot 1=1}\cdot 2^4=30\cdot 16=\boxed{480}\text{ ways in this case.}\]
\[\textit{\underline{\textbf{Solution}}}\]
\[\text{In total, there are }64+192+192+480=\boxed{928}\text{ways.}\]"""
            answer = 928
        elif link == "https://artofproblemsolving.com/wiki/index.php/2018_AIME_I_Problems/Problem_12" and answer == 1:
            solution = r"""The question asks us for the probability that a randomly chosen subset of the set of the first 18 positive integers has the property that the sum of its elements is divisible by 3. Note that the total number of subsets is $2^{18}$ because each element can either be in or not in the subset. To find the probability, we will find the total numbers of ways the problem can occur and divide by $2^{18}$.

To simplify the problem, let’s convert the set to mod 3:

\[U' = \{1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0\}\]
Note that there are six elements congruent to 0 mod 3, 6 congruent to 1 mod 3, and 6 congruent to 2 mod 3. After conversion to mod three, the problem is the same but we’re dealing with much simpler numbers. Let’s apply casework on this converted set based on $S = s(T')$, the sum of the elements of a subset $T'$ of $U'$.

$\textbf{Case 1: }S=0$

In this case, we can restrict the subsets to subsets that only contain 0. There are six 0’s and each one can be in or out of the subset, for a total of $2^{6} = 64$ subsets. In fact, for every case we will have, we can always add a sequence of 0’s and the total sum will not change, so we will have to multiply by 64 for each case. Therefore, let’s just calculate the total number of ways we can have each case and remember to multiply it in after summing up the cases. This is equivalent to finding the number of ways you can choose such subsets without including the 0's. Therefore this case gives us $\boxed{1}$ way.

$\textbf{Case 2: }S= 3$

In this case and each of the subsequent cases, we denote the number of 1’s in the case and the number of two’s in the case as $x, y$ respectively. Then in this case we have two subcases;

$x, y = 3,0:$ This case has $\tbinom{6}{3} \cdot \tbinom{6}{0} = 20$ ways.

$x, y = 1,1:$ This case has $\tbinom{6}{1} \cdot \tbinom{6}{1} = 36$ ways.

In total, this case has $20+36=\boxed{56}$ ways.

$\textbf{Case 3: }S=6$

In this case, there are 4 subcases;

$x, y = 6,0:$ This case has $\tbinom{6}{6} \cdot \tbinom{6}{0} = 1$ way.

$x, y = 4,1:$ This case has $\tbinom{6}{4} \cdot \tbinom{6}{1} = 90$ ways.

$x, y = 2,2:$ This case has $\tbinom{6}{2} \cdot \tbinom{6}{2} = 225$ ways.

$x, y = 0,3:$ This case has $\tbinom{6}{0} \cdot \tbinom{6}{3} = 20$ ways.

In total, this case has $1+90+225+20=\boxed{336}$ ways.

Note that for each case, the # of 1’s goes down by 2 and the # of 2’s goes up by 1. This is because the sum is fixed, so when we change one it must be balanced out by the other. This is similar to the Diophantine equation $x + 2y= S$ and the total number of solutions will be $\tbinom{6}{x} \cdot \tbinom{6}{y}$. From here we continue our casework, and our subcases will be coming out quickly as we have realized this relation.

$\textbf{Case 4: }S=9$

In this case we have 3 subcases:

$x, y = 5,2:$ This case has $\tbinom{6}{5} \cdot \tbinom{6}{1} = 90$ ways.

$x, y = 3,3:$ This case has $\tbinom{6}{3} \cdot \tbinom{6}{3} = 400$ ways.

$x, y = 1,4:$ This case has $\tbinom{6}{1} \cdot \tbinom{6}{4} = 90$ ways.

In total, this case has $90+400+90=\boxed{580}$ ways.

$\textbf{Case 5: } S=12$

In this case we have 4 subcases:

$x, y = 6,3:$ This case has $\tbinom{6}{6} \cdot \tbinom{6}{3} = 20$ ways.

$x, y = 4,4:$ This case has $\tbinom{6}{4} \cdot \tbinom{6}{4} = 225$ ways.

$x, y = 2,5:$ This case has $\tbinom{6}{2} \cdot \tbinom{6}{5} = 90$ ways.

$x, y = 0,6:$ This case has $\tbinom{6}{0} \cdot \tbinom{6}{6} = 1$ way.

Therefore the total number of ways in this case is $20 + 225 + 90 + 1=\boxed{336}$. Here we notice something interesting. Not only is the answer the same as Case 3, the subcases deliver the exact same answers, just in reverse order. Why is that?

We discover the pattern that the values of $x, y$ of each subcase of Case 5 can be obtained by subtracting the corresponding values of $x, y$ of each subcase in Case 3 from 6 ( For example, the subcase 0, 6 in Case 5 corresponds to the subcase 6, 0 in Case 3). Then by the combinatorial identity, $\tbinom{6}{k} = \tbinom{6}{6-k}$, which is why each subcase evaluates to the same number. But can we extend this reasoning to all subcases to save time?

Let’s write this proof formally. Let $W_S$ be the number of subsets of the set $\{1,2,1,2,1,2,1,2,1,2,1,2\}$ (where each 1 and 2 is distinguishable) such that the sum of the elements of the subset is $S$ and $S$ is divisible by 3. We define the empty set as having a sum of 0. We claim that $W_S = W_{18-S}$.

$W_S = \sum_{i=1}^{|D|} \tbinom{6}{x_i}\tbinom{6}{y_i}$ if and only if there exists $x_i, y_i$ that satisfy $0\leq x_i \leq 6$, $0\leq y_i \leq 6$, $x_i + 2y_i = S$, and $D$ is the set of the pairs $x_i, y_i$. This is because for each pair $x_i$, $y_i$ there are six elements of the same residue mod(3) to choose $x_i$ and $y_i$ numbers from, and their value sum must be $S$.

Let all $x_n$, $y_n$ satisfy $x_n = 6-x_i$ and $y_n = 6-y_i$. We can rewrite the equation $x_i+ 2y_i = S \implies -x_i- 2y_i = -S \implies 6-x_i+ 6-2y_i= 12 - S$ $\implies 6-x_i+12-2y_i = 18-S \implies 6-x_i + 2(6-y_i) = 18-S$\[\implies x_n + 2y_n = 18 - S\]
Therefore, since $0\leq x_i, y_i\leq 6$ and $x_n = 6-x_i$ and $y_n = 6-y_i$, $0\leq x_n, y_n\leq 6$. As shown above, $x_n + 2y_n = 18 - S$ and since $S$ and 18 are divisible by 3, 18 -$S$ is divisible by 3. Therefore, by the fact that $W_S = \sum_{i=1}^{|D|} \tbinom{6}{x_i}\tbinom{6}{y_i}$, we have that;

$W_{18-S} = \sum_{n=1}^{|D|} \tbinom{6}{x_n}\tbinom{6}{y_n} \implies W_{18-S} =  \sum_{i=1}^{|D|} \tbinom{6}{6-x_i}\tbinom{6}{6-y_i} \implies W_{18-S} =  \sum_{i=1}^{|D|} \tbinom{6}{x_i}\tbinom{6}{y_i} = W_S$, proving our claim.

We have found our shortcut, so instead of bashing out the remaining cases, we can use this symmetry. The total number of ways over all the cases is $\sum_{k=0}^{6} W_{3k} = 2 \cdot (1+56+336)+580 = 1366$. The final answer is $\frac{2^{6}\cdot 1366}{2^{18}} = \frac{1366}{2^{12}} = \frac{683}{2^{11}}.$

There are no more 2’s left to factor out of the numerator, so we are done and the answer is $\boxed{683}$."""
        answer = 683

        # if we have not stored the problem before, store the link, problem, answer, letter, and solution
        if problem not in train_data:
            train_data[problem] = {
                "link": {link,},  # store links in a set to avoid duplicates
                "problem": problem,
                "answer": answer,
                "letter": letter,
                "solution": [solution,]  # store solutions in a list since we have already removed duplicates
            }
        # if we have stored the problem before, check if the answer and letter are the same,
        # and append the link and solution to the existing problem
        else:
            assert train_data[problem]["answer"] == answer, f"Previously-Stored Answer {train_data[problem]['answer']} != {answer} for {row['link']}"
            # if the previously-stored answer or letter is empty, update it with the current answer or letter
            train_data[problem]["letter"] = letter if train_data[problem]["letter"] == "" else train_data[problem]["letter"]
            train_data[problem]["link"].add(link)
            train_data[problem]["solution"].append(solution)

    # save the train split to a new jsonl file
    print("======================================================")
    print("Saving the train split to ~AIME_2024_I&II_ArtOfProblemSolving.jsonl")
    print("======================================================")
    with open("~AIME_2024_I&II_ArtOfProblemSolving.jsonl", "w", encoding="utf-8") as writer:
        for problem_dict in train_data.values():
            # convert the set of links to a list, so that we can save it a json object
            problem_dict["link"] = list(problem_dict["link"])
            writer.write(json.dumps(problem_dict, ensure_ascii=False) + "\n")


    # select all the rows with distinct "link" column containing "2024_AIME" as the test split
    df_test = df[df["link"].str.contains("2024_AIME")].drop_duplicates(subset=["link"])
    # remove the "letter" column, since AIME problems do not have answer choices
    df_test = df_test.drop(columns=["letter"])

    # the following problems are missing from the dataset
    # 2024 AIME I Problems 1, 2, 5, 7, 11
    # 2024 AIME II Problems 1
    AIME_2024_I_Problem_1 = {
        "link": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems/Problem_1",
        "problem": r"Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.",
        "solution": r"""$\frac{9}{s} + t = 4$ in hours and $\frac{9}{s+2} + t = 2.4$ in hours.

Subtracting the second equation from the first, we get,

$\frac{9}{s} - \frac{9}{s+2} = 1.6$

Multiplying by $(s)(s+2)$, we get

$9s+18-9s=18=1.6s^{2} + 3.2s$

Multiplying by 5/2 on both sides, we get

$0 = 4s^{2} + 8s - 45$

Factoring gives us

$(2s-5)(2s+9) = 0$, of which the solution we want is $s=2.5$.

Substituting this back to the first equation, we can find that $t = 0.4$ hours.

Lastly, $s + \frac{1}{2} = 3$ kilometers per hour, so

$\frac{9}{3} + 0.4 = 3.4$ hours, or $\framebox{204}$ minutes.""",
        "answer": 204
    }
    AIME_2024_I_Problem_2 = {
        "link": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems/Problem_2",
        "problem": r"Real numbers $x$ and $y$ with $x,y>1$ satisfy $\log_x(y^x)=\log_y(x^{4y})=10.$ What is the value of $xy$?",
        "solution": r"""By properties of logarithms, we can simplify the given equation to $x\log_xy=4y\log_yx=10$. Let us break this into two separate equations:

\[x\log_xy=10\]\[4y\log_yx=10.\]We multiply the two equations to get:\[4xy\left(\log_xy\log_yx\right)=100.\]
Also by properties of logarithms, we know that $\log_ab\cdot\log_ba=1$; thus, $\log_xy\cdot\log_yx=1$. Therefore, our equation simplifies to:

\[4xy=100\implies xy=\boxed{25}.\]""",
        "answer": 25
    }
    AIME_2024_I_Problem_5 = {
        "link": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems/Problem_5",
        "problem": r"Rectangles $ABCD$ and $EFGH$ are drawn such that $D,E,C,F$ are collinear. Also, $A,D,H,G$ all lie on a circle. If $BC=16,$ $AB=107,$ $FG=17,$ and $EF=184,$ what is the length of $CE$?",
        "solution": r"""We find that\[\angle GAB = 90-\angle DAG = 90 - (180 - \angle GHD) = \angle DHE.\]
Let $x = DE$ and $T = FG \cap AB$. By similar triangles $\triangle DHE \sim \triangle GAT$ we have $\frac{DE}{EH} = \frac{GT}{AT}$. Substituting lengths we have $\frac{x}{17} = \frac{16 + 17}{184 + x}.$ Solving, we find $x = 3$ and thus $CE = 107 - 3 = \boxed{104}.$""",
        "answer": 104
    }
    AIME_2024_I_Problem_7 = {
        "link": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems/Problem_7",
        "problem": r"Find the largest possible real part of\[(75+117i)z+\frac{96+144i}{z}\]where $z$ is a complex number with $|z|=4$.",
        "solution": r"""First, recognize the relationship between the reciprocal of a complex number $z$ with its conjugate $\overline{z}$, namely:

\[\frac{1}{z} \cdot \frac{\overline{z}}{\overline{z}} = \frac{\overline{z}}{|z|^2} = \frac{\overline{z}}{16}\]
Then, let $z = 4(\cos\theta + i\sin\theta)$ and $\overline{z} = 4(\cos\theta - i\sin\theta)$.

\begin{align*} Re \left ((75+117i)z+\frac{96+144i}{z} \right) &= Re\left ( (75+117i)z + (6+9i)\overline{z}    \right ) \\                                                &= 4 \cdot Re\left ( (75+117i)(\cos\theta + i\sin\theta) + (6+9i)(\cos\theta - i\sin\theta)    \right ) \\                                                &= 4 \cdot (75\cos\theta - 117\sin\theta + 6\cos\theta + 9\sin\theta) \\                                                &= 4 \cdot (81\cos\theta - 108\sin\theta) \\                                                &= 4\cdot 27 \cdot (3\cos\theta - 4\sin\theta) \end{align*}
Now, recognizing the 3 and 4 coefficients hinting at a 3-4-5 right triangle, we "complete the triangle" by rewriting our desired answer in terms of an angle of that triangle $\phi$ where $\cos\phi = \frac{3}{5}$ and $\sin\phi = \frac{4}{5}$

\begin{align*} 4\cdot 27 \cdot(3\cos\theta - 4\sin\theta) &= 4\cdot 27 \cdot 5 \cdot (\frac{3}{5}\cos\theta - \frac{4}{5}\sin\theta) \\                                                &= 540 \cdot (\cos\phi\cos\theta - \sin\phi\sin\theta) \\                                                &= 540 \cos(\theta + \phi) \end{align*}
Since the simple trig ratio is bounded above by 1, our answer is $\boxed{540}$""",
        "answer": 540
    }
    AIME_2024_I_Problem_11 = {
        "link": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems/Problem_11",
        "problem": r"Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability that the octagon can then be rotated so that all of the blue vertices end up at positions where there were originally red vertices is $\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. What is $m+n$?",
        "solution": r"""Let $r$ be the number of red vertices and $b$ be the number of blue vertices, where $r+b=8$. By the Pigeonhole Principle, $r\geq{b} \Longrightarrow b\leq4$ if a configuration is valid.

We claim that if $b\leq3$, then any configuration is valid. We attempt to prove by the following:

If there are\[b\in{0,1,2}\]vertices, then intuitively any configuration is valid. For $b=3$, we do cases:

If all the vertices in $b$ are non-adjacent, then simply rotating once in any direction suffices. If there are $2$ adjacent vertices, then WLOG let us create a set $\{b_1,b_2,r_1\cdots\}$ where the third $b_3$ is somewhere later in the set. If we assign the set as $\{1,2,3,4,5,6,7,8\}$ and $b_3\leq4$, then intuitively, rotating it $4$ will suffice. If $b_3=5$, then rotating it by 2 will suffice. Consider any other $b_3>5$ as simply a mirror to a configuration of the cases.

Therefore, if $b\leq3$, then there are $\sum_{i=0}^{3}{\binom{8}{i}}=93$ ways. We do count the [i]degenerate[/i] case.

Now if $b=4$, we do casework on the number of adjacent vertices. 0 adjacent: $\{b_1,r_1,b_2,r_2\cdots{r_4}\}$. There are 4 axes of symmetry so there are only $\frac{8}{4}=2$ rotations of this configuration.

1 adjacent: WLOG $\{b_1,b_2\cdots{b_3}\cdots{b_4}\}$ where $b_4\neq{8}$. Listing out the cases and trying, we get that $b_3=4$ and $b_4=7$ is the only configuration. There are $8$ ways to choose $b_1$ and $b_2$ and the rest is set, so there are $8$ ways.

2 adjacent: We can have WLOG $\{b_1,b_2\cdots{b_3},b_4\}$ or $\{b_1,b_2,b_3\cdots\}$ where $b_4\neq{8}$. The former yields the case $b_3=5$ and $b_4=6$ by simply rotating it 2 times. The latter yields none. There are 2 axes of symmetry so there are $\frac{8}{2}=4$ configurations.

3 adjacent: WLOG $\{b_1,b_2,b_3,b_4\cdots\}$ which intuitively works. There are $8$ configurations here as $b_1$ can is unique.

In total, $b=4$ yields $2+8+4+8=22$ configurations.

There are $22+93=115$ configurations in total. There are $2^8=256$ total cases, so the probability is $\frac{115}{256}$. Adding them up, we get $115+256=\boxed{371}$.""",
        "answer": 371
    }
    AIME_2024_II_Problem_1 = {
        "link": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems/Problem_1",
        "problem": r"Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.",
        "solution": r"""We know that there are 195 diamond rings, 367 golf clubs, and 562 garden spades, so we can calculate that there are $195+367+562=1124$ items, with the exclusion of candy hearts which is irrelevant to the question. There are 437 people who owns 2 items, which means 1 item since candy hearts are irrelevant, and there are 234 people who own 2 items plus a bag of candy hearts, which means that the 234 people collectively own $234*2=468$ items. We can see that there are $1124-437-468=219$ items left, and since the question is asking us for the people who own 4 items, which means 3 items due to the irrelevance of candy hearts, we simply divide 219 by 3 and get $219/3=\boxed{073}$.""",
        "answer": 73
    }

    # the following problems are incomplete in the dataset
    # 2024 AIME II Problems 9
    AIME_2024_II_Problem_9 = {
        "link": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems/Problem_9",
        "problem": r"""There is a collection of $25$ indistinguishable white chips and $25$ indistinguishable black chips. Find the number of ways to place some of these chips in the $25$ unit cells of a $5\times5$ grid such that:

each cell contains at most one chip
all chips in the same row and all chips in the same column have the same colour
any additional chip placed on the grid would violate one or more of the previous two conditions.""",
        "solution": r"The problem says 'some', so not all cells must be occupied. We start by doing casework on the column on the left. There can be 5,4,3,2, or 1 black chip. The same goes for white chips, so we will multiply by 2 at the end. There is $1$ way to select $5$ cells with black chips. Because of the 2nd condition, there can be no white, and the grid must be all black- $1$ way . There are $5$ ways to select 4 cells with black chips. We now consider the row that does not contain a black chip. The first cell must be blank, and the remaining $4$ cells have $2^4-1$ different ways($-1$ comes from all blank). This gives us $75$ ways. Notice that for 3,2 or 1 black chips on the left there is a pattern. Once the first blank row is chosen, the rest of the blank rows must be ordered similarly. For example, with 2 black chips on the left, there will be 3 blank rows. There are 15 ways for the first row to be chosen, and the following 2 rows must have the same order. Thus, The number of ways for 3,2,and 1 black chips is $10*15$, $10*15$, $5*15$. Adding these up, we have $1+75+150+150+75 = 451$. Multiplying this by 2, we get $\boxed{902}$.",
        "answer": 902
    }

    # delete the incomplete problems from the dataframe
    df_test = df_test[~df_test["link"].str.contains("2024_AIME_II_Problems/Problem_9")]

    # add the missing and update the incomplete problems to the dataframe
    missing_problems = [
        AIME_2024_I_Problem_1,
        AIME_2024_I_Problem_2,
        AIME_2024_I_Problem_5,
        AIME_2024_I_Problem_7,
        AIME_2024_I_Problem_11,
        AIME_2024_II_Problem_1,
        AIME_2024_II_Problem_9,
    ]
    missing_problems_df = pd.DataFrame(missing_problems, columns=["link", "problem", "solution", "answer"])
    df_test = pd.concat([df_test, missing_problems_df], ignore_index=True)

    # convert the "answer" column to integer
    df_test["answer"] = df_test["answer"].astype(int)

    # sort by "link" column
    df_test = df_test.sort_values(by="link")
    # assert that there are 30 problems in the test split
    assert len(df_test) == 30, f"Expected 30 problems in the test split, but got {len(df_test)}"

    # save the test split to a new jsonl file
    print("======================================================")
    print("Saving the test split to AIME_2024_I&II_ArtOfProblemSolving.jsonl")
    print("======================================================")
    with open("AIME_2024_I&II_ArtOfProblemSolving.jsonl", "w", encoding="utf-8") as writer:
        for _, row in df_test.iterrows():
            writer.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
