# Heat and Mass Transfer Codes

I'm looking to develop an easy to understand and easy to use Python tool for heat and mass transfer simulation. 
In the course of my Cambridge study there has been similar codes being demonstrated, but mostly in 1D and/or in
Cartesian coordinates. I've recently come across a few questions asking for then temperature profile of a 
circular cross section. Even though the heat equations are easy to solve, I look forward to see how, without full
knowledge on simulation codes, my heat transfer codes would perform.

## Day 1, 2025 Feb 2nd
The first few hours of developing it only led to instabilities, where there are large stripes with really high
temperature and really low temperature close together. The intuition is that either 1) updating on the original
temoerature grid does more harm then good 2) averaging of nearby cells is needed

## Day 2, Feb 3rd Update
Update Feb 3rd. polar coordinate's instability solved by 'averaging temperature' in the theta direction using
convolutions. This is not ideal and the focued is turned to Cartesian coordinates.

The cartesian coordinate simulator worked much better. Although it is visible that the propagation at the boundary 
is much slower than that away from the boundary - this could be the case that it's being updated with a different 
law. Generally speaking, the simulator is slow and would only approximate the real 2D solution after O(1000) 
iterations.

The Q2e code covers a semicircle, heated at the diameter to a temperature higher than the environment. The arc is
kept at a constant temperature (environment). Only conduction is modeled.

Another simulator based on the conduction only one is the conduction + convection. It yields largely the same 
result as the conduction only simulation. The difference that could be visible is that it has a non-uniform 
temperature on the arc. (Q2f code).

## Remarks
If anyone's got any comments, please let me know! This remains an individual project for heat conduction for now.
