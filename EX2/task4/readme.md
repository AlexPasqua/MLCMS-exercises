### After having built _vadere_
- Added pedestrians in their correct initial groups:
  - file: [`SIRGroupModel.java`](SIRGroupModel.java)
  - to be placed in: `org.vadere.simulator.models.groups.sir`
  - modified lines (might be outdated): 137-139

### To fix the group color
- Place [SimulationModel](SimulationModel.java) in `vadere/VadereGui/src/org/vadere/gui/components/model`
- Added 3 color attributes in [SimulationModel](SimulationModel.java)
- Create getter methods to get the color of different states (infective / susceptible / recovered)
- Correct initialization of the hashmap to contain the colors (lines 34-37)
- In [SimulationModel](SimulationModel.java)'s constructor, change `this.config.setAgentColoring(AgentColoring.TARGET);` to `this.config.setAgentColoring(AgentColoring.GROUP);`