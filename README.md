# **Airdrop campaigns as an incentive mechanism for early adoption of web3 applications**

**Steps:**

Add on-chain data in the format Date: ***dd/mm/yyyyy*** and value of **TVL, DAU,Volume** in the corresponding directory

***pip3 install pandas numpy scipy statsmodels***

***python3 analysis.py***

Retrieve the result from the folder result

**Methodology:**

The use cases of web3 applications can be grouped into the categories of Decentralized Finance (DeFi), Decentralized exchanges (DEX) and Bridges as well as decentralized social media (SocialFi). Each type of protocol has one primary indicator that estimates the development and adoption of the platform. The impact of the Airdrop treatment is estimated by applying a Multi-Regime Interrupted Time Series model with the following key metrics Total Value Locked (TVL), Daily Transaction Volume, Daily Active Users (DAU).

For each protocol type i (DeFi, DEX, SocialFi), ranging from 30 days before to 30 days after the token allocation, the models are defined as follows:

DeFi Protocols (Total Value Locked)

***DeFi_it = α0i + α1iTt + α2iXt + α3iXtTt + α4iTt + α5itXt + δ1*MCt + δ2FGIt +δ3S&P500t + εit (1)***


---

DEX/Bridge Protocols (Daily Transaction Volume)

***DEXit = β0i + β1iTt + β2iXt + β3iXtTt + β4iTt + β5itXt + δ1MCt + δ2FGIt + δ3S&P500t + εit (2)***

SocialFi Protocols (Daily Active Users)

***SocialFi_it = γ0i + γ1iTt + γ2iXt + γ3iXtTt + γ4iTt + γ5itXt + δ1MCt + δ2FGIt + δ3S&P500t + εit (3)***

Where:

***DeFi_it, DEXit, SocialFi_it***  is the metric value for protocol type **i** at time **t**

***Tt***  is the time variable,
centered at the airdrop date (ranges from -30 to +30)

***Xt*** is the intervention variable
(0 for pre-airdrop, 1 for post-airdrop including airdrop day)

***α0i, β0i, γ0i*** are the intercept

***α1i, β1i, γ**1i* are the pre-intervention slopes for protocol type i

***α2i, β2i, γ2i*** are the level changes after the airdrop for protocol type i

***α3i, β3i, γ3**i* are the slope changes after the airdrop for protocol type i

**α4i, β4i, γ4i** are the coefficients for the time trend

**α5i, β5i, γ5i** are the coefficients for the interaction between time trend and intervention

***MCt*** is the market capitalization of all altcoins at time t

**FGIt** is the Fear and Greed Index at time t

***δ1***  is the coefficient for market capitalization

***δ2***  is the coefficient for the Fear and Greed Index

***εi***  is the remaining error term
