"""Save the team/venue table the college pipeline needs -> /tmp/cfb/data/team_info_<year>.pkl"""
import sys; sys.path.insert(0, '/home/claude/cfbpredict')
from rds_reader import read_rds
COLS = ['team_id','school','abbreviation','conference','classification','venue_name','city','state','timezone','latitude','longitude','elevation','capacity','grass','dome','color','logo']
for y in (2025, 2026):
    t = read_rds(f'/home/claude/sportsdataverse/cfbfastr-data/team_info/rds/cfb_team_info_{y}.rds')
    t[COLS].to_pickle(f'/tmp/cfb/data/team_info_{y}.pkl'); print(y, len(t))
