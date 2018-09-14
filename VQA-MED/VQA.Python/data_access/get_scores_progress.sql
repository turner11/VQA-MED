
SELECT models.id as model_id 
	   , s.bleu
	   , s.wbss 
	   , models.trainable_parameter_count 
	   ,models.notes
	      , bleu + wbss as combined
	  -- ,models.loss_function
	   --,models.activation
	   ,"('"||models.loss_function || "', '"|| models.activation ||"')," as str 
	  -- , models.loss_function || ' | '|| models.activation || ' | '|| models.trainable_parameter_count || ' | '|| printf("%.4f",bleu) || ' | '|| printf("%.4f",wbss) || ' | '|| model_id || ' |' as markdown
	   --,models.*	  
FROM Models 
	LEFT JOIN scores as s on models.id = s.model_id
--WHERE bleu_id IS NOT NULL OR wb_id IS NOT NULL 
ORDER BY models.id DESC
	