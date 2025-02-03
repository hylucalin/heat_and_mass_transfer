# Heat and Mass Transfer Codes

I'm looking to develop an easy to understand and easy to use Python tool for heat and mass transfer simulation. 
In the course of my Cambridge study there has been similar codes being demonstrated, but mostly in 1D and/or in
Cartesian coordinates. I've recently come across a few questions asking for then temperature profile of a 
circular cross section. Even though the heat equations are easy to solve, I look forward to see how, without full
knowledge on simulation codes, my heat transfer codes would perform.

The first few hours of developing it only led to instabilities, where there are large stripes with really high
temperature and really low temperature close together. The intuition is that either 1) updating on the original
temoerature grid does more harm then good 2) averaging of nearby cells is needed

If anyone's got any comments, please let me know! This remains an individual project for heat conduction for now.
