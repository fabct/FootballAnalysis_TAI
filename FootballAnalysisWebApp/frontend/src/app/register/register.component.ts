import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';

@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css']
})
export class RegisterComponent {
  username: string = '';
  password: string = '';
  loginError: string = '';
  phone: string = '';
  constructor(private http: HttpClient) { }

  onSubmit(): void {
    // Vérification des champs username et password avant d'envoyer la requête
    if (this.username && this.password && this.phone) {
      if (/^\d{9}$/.test(this.phone)) {

        const loginData = {
          user: this.username,
          password: this.password,
          telephone: this.phone
        };

        // Envoi de la requête POST vers l'URL fournie
        this.http.post<any>('https://lorenzo-geano-etu.pedaweb.univ-amu.fr/extranet/webservice/CRM/Adduser.php', loginData)
          .subscribe(
            response => {
              // Gestion de la réponse en cas de succès
              alert(response) // Affichage de la réponse dans la console
            window.location.reload();
            },
            error => {
              // Gestion de l'erreur en cas d'échec de la requête
              console.error(error); // Affichage de l'erreur dans la console
              this.loginError = error.error.error; // Message d'erreur pour l'utilisateur
            }
          );
      } else {
        this.loginError = 'Veuillez remplir un vrai numéro de téléphone français.(aucun espace entre les chiffres)'; // Message si les champs ne sont pas remplis
      }
    } else {
      this.loginError = 'Veuillez remplir tous les champs.'; // Message si les champs ne sont pas remplis
    }
  }
}

