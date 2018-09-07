SELECT s.* 
	   , bleu + wbss as combined
	   ,models.loss_function
	   ,models.activation
	   ,"('"||models.loss_function || "', '"|| models.activation ||"')," as str 
	   --,models.*	  
FROM SCORES s
	LEFT JOIN (
	SELECT bs.model_id as bleu_id
	FROM SCORES as bs 
	ORDER BY bleu DESC 
	LIMIT 5
	) as topB on s.model_id = topB.bleu_id
	
	LEFT JOIN (
	SELECT wb.model_id as wb_id
	FROM SCORES as wb
	ORDER BY wbss DESC 
	LIMIT 5
	) as topW on s.model_id = topW.wb_id
	
	INNER JOIN models on models.id = s.model_id
WHERE bleu_id IS NOT NULL OR wb_id IS NOT NULL 
ORDER BY wbss DESC, bleu DESC
	