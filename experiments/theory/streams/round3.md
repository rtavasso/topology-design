# Round 3 — where the sharp gate's extra gain comes from (EXPLORATORY)

Prediction in `round3.py`.

- median gain, soft gate at tau = 0.05: 7.75; with the itinerary frozen: 4.33; single streams alone (round 2): 5.17
- inputs far from every boundary (22%): soft 5.58, frozen 5.58
- inputs near a boundary (78%): soft 8.87, frozen 4.00
- itineraries that use one stream throughout: 34% of inputs, frozen gain 5.77; itineraries that switch: frozen gain 3.90
