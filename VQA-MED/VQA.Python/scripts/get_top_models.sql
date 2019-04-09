SELECT models.id as model_id
		,models.class_strategy
		,s.score
		,s.category
		,s.Evaluation
		,s.model_location
		,s.notes
FROM models 
LEFT JOIN v_scores as s ON models.id = s.model_id
WHERE 1=1
	--AND category = 'Abnormality'
	--AND models.id > 65
	--AND models.id = 72
	--AND evaluation = 'wbss'
	AND model_id <> 47


GROUP BY category, Evaluation	
	
--ORDER BY s.score IS NOT NULL
--ORDER BY s.model_id DESC

ORDER BY max(score) DESC

